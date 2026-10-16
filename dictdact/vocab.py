"""Custom vocabulary — the speaker's names, products and tools, written exactly the way they
want no matter how the speech model heard them (a Wispr-Flow-style dictionary).

Three things use it:
- the Whisper prompt gets the terms as a tail, which biases spelling at decode time;
- the LLM formatter gets a glossary line, so it maps "ack me labs" → "Acme Labs" in context;
- deterministic replacements run on the raw transcript AND on the formatted text, so a term
  is right even when both models miss it (this is the part that makes it reliable).

The file is ~/.config/dictdact/vocabulary.json — plain JSON, hand-editable:
    {"terms": [{"term": "Acme Labs", "spoken": ["ack me labs", "acne labs"], "note": "the company"}]}
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CONFIG_DIR

log = logging.getLogger("dictdact.vocab")

VOCAB_PATH = CONFIG_DIR / "vocabulary.json"

# One lock per file: a per-path threading.RLock serializes THREADS in this process (the HTTP
# server handles requests concurrently; re-entrant so save() may run inside a transaction) and
# flock serializes PROCESSES (the window, the CLI and the running app may all do load → modify
# → save on the same file). The flock is taken by the thread holding the RLock at depth 0.
_registry_guard = threading.Lock()
_path_locks: dict[str, threading.RLock] = {}
_lock_depth: dict[str, int] = {}
_lock_fds: dict[str, int] = {}


def _path_lock(key: str) -> threading.RLock:
    with _registry_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = _path_locks[key] = threading.RLock()
        return lock


@contextlib.contextmanager
def vocab_lock(path: Path = VOCAB_PATH):
    key = str(path)
    lock = _path_lock(key)
    lock.acquire()                    # blocks other threads; re-entrant for this one
    try:
        depth = _lock_depth.get(key, 0)
        if depth == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path.with_name(path.name + ".lock")), os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            _lock_fds[key] = fd
        _lock_depth[key] = depth + 1
        try:
            yield
        finally:
            _lock_depth[key] -= 1
            if _lock_depth[key] == 0:
                fd = _lock_fds.pop(key)
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
    finally:
        lock.release()

# Whisper keeps only the last 223 prompt tokens; a term is ~3 tokens. Leave room for the sentence.
MAX_PROMPT_TERMS = 40

# File format version. 1 = the first seed list; 2 = seed variants that also read as everyday
# words were retired from the seeds — a v1 file on disk is migrated once and rewritten.
# Each retired spelling is removed ONLY from the seeded term it shipped on; the same spelling
# on a term the user authored is theirs and stays.
VOCAB_VERSION = 3        # 3 = the fuzzy flag exists; a v2 file gets each untouched seed's flag
# (seed term, retired spelling): v1 shipped "codecs" as a variant of Codex, but it is an ordinary
# word, so replacing it everywhere was wrong.
RETIRED_SEED_VARIANTS = frozenset((("codex", "codecs"),))
RETIRED_VARIANTS = frozenset(v for _, v in RETIRED_SEED_VARIANTS)
# The exact v1 spoken lists of those seeds. Provenance by equality: a term whose list still
# reads exactly like the seed was never touched by the user and is migrated; a list the user
# changed in any way (added, removed, reordered) is theirs and is left alone.
_V1_SEED_SPOKEN = {
    "codex": ["codecs"],
}


@dataclass
class Term:
    term: str                                   # written exactly like this
    spoken: list[str] = field(default_factory=list)   # what the speech model tends to write instead
    note: str = ""                              # short hint for the LLM ("the company", "the fitness app")
    fuzzy: bool = False                         # also catch sound-alike spellings (Hirox → Hyrox): OPT-IN — on for
                                                # seeded product names, for a suggestion the user accepted, or when
                                                # they tick the box / pass --fuzzy; never switched on by a fix
    learned: list[str] = field(default_factory=list)  # spellings learned from the user's fixes — applied only
                                                      # where a name can stand (capitalized mid-sentence), never
                                                      # to an ordinary lowercase word

    @classmethod
    def from_dict(cls, d: dict) -> "Term":
        term = str(d.get("term", "")).strip()
        spoken = [str(s).strip() for s in (d.get("spoken") or []) if str(s).strip()]
        learned = [str(s).strip() for s in (d.get("learned") or []) if str(s).strip()]
        return cls(term=term, spoken=spoken, note=str(d.get("note", "") or "").strip(),
                   fuzzy=bool(d.get("fuzzy", False)), learned=learned)


# --- sound-alike matching -----------------------------------------------------------------
# Whisper hears a name it does not know and writes the nearest thing it does: Hirox, Hierox,
# Hyrux and Irox are all "Hyrox" said by the same person. A term is compared with the words of
# a transcript through a coarse Spanish/English sound key (consonant skeleton after folding the
# usual confusions) plus a similarity ratio, so the user adds "Hyrox" once and every spelling
# Whisper invents for it is caught — the "learn how I say things" part that no list can cover.
_SOUND_MAP = (("ph", "f"), ("ck", "k"), ("qu", "k"), ("sh", "x"), ("ch", "x"), ("ll", "y"), ("gue", "ge"),
              ("gui", "gi"), ("ce", "se"), ("ci", "si"), ("c", "k"), ("z", "s"), ("v", "b"), ("w", "u"),
              ("h", ""), ("y", "i"), ("x", "ks"))
FUZZY_MIN_CHARS = 4
FUZZY_RATIO = 0.80              # a sound-alike by spelling alone (Superbase / Supabase, Hicksfield / Higgsfield)
FUZZY_KEY_RATIO = 0.60          # same sound key AND at least this close (Simon / Simone yes, Luna / Simone no)
FUZZY_MARGIN = 0.08             # the best term must beat the runner-up by this much (no first-declared-wins)


def sound_key(s: str) -> str:
    s = re.sub(r"[^a-z]", "", _fold(s))
    for a, b in _SOUND_MAP:
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[0] + re.sub(r"[aeiou]", "", s[1:]) if s else s


def similarity(candidate: str, term: str) -> float:
    """0..1 score of ``candidate`` as ``term`` misheard: the spelling ratio, lifted when the two
    share a sound key. 0 when the candidate simply IS the term or is too short to judge."""
    a, b = _fold(candidate), _fold(term)
    a_c, b_c = re.sub(r"[^a-z0-9]", "", a), re.sub(r"[^a-z0-9]", "", b)
    if not a_c or not b_c or len(a_c) < FUZZY_MIN_CHARS or a == b:
        return 0.0
    if a_c == b_c:
        return 1.0                              # same letters, different spacing / casing: "Wisprflow" IS "Wispr Flow"
    import difflib

    ratio = difflib.SequenceMatcher(None, a_c, b_c).ratio()
    if ratio >= FUZZY_RATIO:
        return ratio
    if sound_key(a) == sound_key(b) and ratio >= FUZZY_KEY_RATIO:
        return max(ratio, FUZZY_RATIO)          # same sound: as good as a spelling match
    return 0.0


def sounds_like(candidate: str, term: str) -> bool:
    """True when ``candidate`` is plausibly ``term`` misheard."""
    return similarity(candidate, term) > 0.0


# A token keeps its inner dots and hyphens ("Acmetunes.me", "GPT-Image", "acmelabs.io") so a dotted
# term is one unit — never "Acmetunes" + ".me" (which once produced "Acmetunes.me.me").
_TOKEN_RX = re.compile(r"[A-Za-zÀ-ÿ0-9](?:[A-Za-zÀ-ÿ0-9'\-.]*[A-Za-zÀ-ÿ0-9])?")
_SENTENCE_END_RX = re.compile(r"[.!?¿¡\n]\s*$")
# What also starts a "sentence" for the capitalization signal: a bullet, a numbered item, an opening
# quote or bracket — the formatter emits bullet lists, and "- Codecs" must not read as a name.
_ITEM_START_RX = re.compile(r"(?:^|\n)\s*(?:[-•*–]|\d+[.)])\s*$|[\"“«(\[]\s*$")
_WINDOW_BAD_RX = re.compile(r"[.,;:!?¿¡\"“”«»()\[\]\n]")     # never inside a multi-word window


def _name_like(token: str, preceding: str) -> bool:
    """Whisper capitalizes the names it does not know; a capitalized word inside a sentence
    (not after . ! ? or at the very start, not a bullet or a quote opener) is the signal that
    keeps fuzzy matching away from ordinary words — "codecs" stays "codecs", "Codecs are…" at
    sentence start stays too."""
    if not token[0].isupper():
        return False
    if not preceding.strip() or _SENTENCE_END_RX.search(preceding) or _ITEM_START_RX.search(preceding):
        return token.isupper() and len(token) >= FUZZY_MIN_CHARS      # "WOOP" — all caps counts anywhere
    return True


_SYSTEM_WORDS: frozenset[str] | None = None


def system_words(path: str = "/usr/share/dict/words") -> frozenset[str]:
    """macOS ships an English word list; a capitalized word that is also a plain English word
    ("Cloud" in "Google Cloud", "Bible", "Neon", "Simon") is never a misheard name. Empty when
    absent — and then sound-alike matching stays OFF (fail closed)."""
    global _SYSTEM_WORDS
    if _SYSTEM_WORDS is None:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                _SYSTEM_WORDS = frozenset(w.strip().lower() for w in f if len(w.strip()) >= 3)
        except OSError:
            _SYSTEM_WORDS = frozenset()
    return _SYSTEM_WORDS


def _is_plain_word(folded: str, protected: frozenset[str]) -> bool:
    """The word, or its singular / possessive base, is a protected (ordinary) word."""
    if folded in protected:
        return True
    for suffix in ("s", "es", "'s"):
        if folded.endswith(suffix) and folded[: -len(suffix)] in protected:
            return True                          # "Codecs" is "codec" + s
    return False


# Example seeds: spellings Whisper commonly produces for these terms. The user edits the list
# from the window / the CLI / the file.
# A spoken variant is replaced EVERYWHERE it appears, so the defaults carry no everyday
# phrase; an ambiguous one is left to the LLM glossary, which maps it in context — add it here
# yourself if you never use it as a word.
DEFAULT_TERMS: list[Term] = [
    Term("DictDact", ["dictac", "dict dact", "dic dac", "dict-dact", "dick dack", "dictact"], "this dictation app", fuzzy=True),
    Term("Wispr Flow", ["whisper flow", "whisperflow", "whisper flo", "whisperfo", "wisper flow", "whisper flor"], fuzzy=True),
    Term("TestFlight", ["test flight"], fuzzy=True),
    Term("Supabase", ["superbase", "supa base", "super base"], fuzzy=True),
    Term("ManyChat", ["manichat", "many chat", "manny chat"], fuzzy=True),
    Term("Hyrox", ["hirox", "hierox", "high rocks", "hyrocks"], fuzzy=True),
    Term("Higgsfield", ["higgs field", "hicksfield", "hicks field", "higsfield"], fuzzy=True),
    Term("WHOOP", ["whoop", "woop"], "the wearable", fuzzy=True),
    Term("Codex", [], "OpenAI's coding agent", fuzzy=False),
    Term("Claude Code", ["cloud code", "clod code"], fuzzy=False),
    Term("Claude", ["clod"], "Anthropic's assistant", fuzzy=False),
    Term("MLX", ["M L X"], "Apple's machine-learning framework", fuzzy=False),
    Term("MiniMax", ["minimax", "mini max"], fuzzy=True),
    Term("ComfyUI", ["comfy UI", "comfy you I", "comfy U I"], fuzzy=True),
    Term("Cloudflare", ["cloud flare"], fuzzy=True),
    Term("Vercel", ["versel", "bercel"], fuzzy=True),
    Term("Chatterbox", ["chatter box"], fuzzy=False),
]

_SEP_CHARS = r"[\s\-.,·]"   # what may sit between the pieces of a spoken variant ("Dict, Dact", "M-L-X")
_ACCENTS = {"a": "[aáàâä]", "e": "[eéèêë]", "i": "[iíìîï]", "o": "[oóòôö]", "u": "[uúùûü]", "n": "[nñ]", "c": "[cç]"}


def _fold(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _piece_rx(piece: str) -> str:
    """A word of a spoken variant as a regex: case-insensitive, accent-tolerant on the input side
    ("agencia" matches "agencía"), literal otherwise."""
    out = []
    for ch in _fold(piece):
        out.append(_ACCENTS.get(ch, re.escape(ch)))
    return "".join(out)


def _variant_rx(variant: str) -> str:
    """"ack me labs" → ack<sep+>me<sep+>labs ; "M L X" → m<sep*>l<sep*>x.
    Between two single letters (a spelled-out acronym) the separator is optional, so
    "M-L-X", "M.L.X" and "MLX" all match; between whole words at least one is
    required, so "acme labs" cannot swallow "acmelabs.io"."""
    parts = [p for p in re.split(r"([\s\-.,·]+)", variant.strip()) if p]
    pieces = parts[::2]
    seps = parts[1::2]
    if not pieces or any(not p.strip() for p in pieces):
        return ""
    out = [_piece_rx(pieces[0])]
    for k, (prev, cur) in enumerate(zip(pieces, pieces[1:])):
        if len(prev) == 1 and len(cur) == 1:
            out.append(_SEP_CHARS + "*")         # "M-L-X", "M.L.X", "MLX"
        elif "." in (seps[k] if k < len(seps) else ""):
            out.append(_SEP_CHARS + "+")         # written with a dot ("acme labs.io"): a dot is fine
        else:
            out.append(r"[\s\-,·]+")             # whole words: a space, hyphen or comma — never a period
        out.append(_piece_rx(cur))               # ("Whisper. Flow" is two sentences, not the term)
    return r"(?<!\w)" + "".join(out) + r"(?!\w)"


def _stylized(term: str) -> bool:
    """A term whose casing is not plain Capitalized/lower ("MLX", "ManyChat", "WHOOP",
    "Wispr Flow") — or that carries a dot or a digit ("acmelabs.io", "Acmetunes.me") — gets its
    own spelling as an implicit variant, so "mlx" and "Wispr flow" are fixed too. Plain names
    ("Codex", "Render") are not — capitalizing every "render" in a sentence would be wrong."""
    return (term != term.capitalize() and term != term.lower()) or bool(re.search(r"[.\d]", term))


class Vocabulary:
    def __init__(self, terms: list[Term] | None = None):
        self.terms: list[Term] = [t for t in (terms if terms is not None else []) if t.term]
        self._rules: list[tuple[str, str]] = []       # (variant regex, term) — longest variant first
        self._rx: re.Pattern | None = None            # one alternation of all of them
        self._by_group: dict[str, str] = {}
        # Folded words fuzzy matching must never touch: the system's English word list plus
        # the words the speaker uses in lowercase (added from their history by the pipeline).
        # No word list → no sound-alike matching at all (fail closed).
        self.protected: frozenset[str] = system_words()
        self._compile()

    # -- persistence ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> "Vocabulary":
        with vocab_lock(path):
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    version = int(data.get("version", 1) or 1)
                    raw_terms = list(data.get("terms", []))
                    if version < 3:
                        # The fuzzy flag did not exist yet: a seeded term that still reads like the seed
                        # takes the seed's flag (people, characters and acronyms ship with it off).
                        seeds = {_fold(t.term): t for t in DEFAULT_TERMS}
                        for d in raw_terms:
                            if isinstance(d, dict) and "fuzzy" not in d:
                                seed = seeds.get(_fold(str(d.get("term", ""))))
                                d["fuzzy"] = seed.fuzzy if seed is not None else False   # a user's term: opt-in
                    terms = [Term.from_dict(d) for d in raw_terms]
                except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
                    log.warning("vocabulary %s unreadable (%s) — using the built-in list", path, e)
                    return cls(list(DEFAULT_TERMS))
                v = cls(terms)
                if version < VOCAB_VERSION:
                    dropped = v._migrate(version)
                    log.info("vocabulary migrated v%d → v%d (%d retired variant(s) removed)", version, VOCAB_VERSION, dropped)
                    try:
                        v.save(path)
                    except OSError as e:
                        log.warning("could not rewrite %s after migration: %s", path, e)
                return v
            v = cls(list(DEFAULT_TERMS))
            try:
                v.save(path)                      # write the defaults so they can be edited
            except OSError as e:
                log.warning("could not write %s: %s", path, e)
            return v

    def _migrate(self, from_version: int) -> int:
        """v1 → v2: the seed variants that also read as everyday phrases are retired from the
        seeded terms they shipped on — and only while that term's spoken list is still exactly
        the seed's. A list the user touched in any way is theirs and stays as it is."""
        dropped = 0
        if from_version < 2:
            for t in self.terms:
                seed = _V1_SEED_SPOKEN.get(_fold(t.term))
                if seed is None or [_fold(s) for s in t.spoken] != seed:
                    continue
                keep = [s for s in t.spoken if (_fold(t.term), _fold(s)) not in RETIRED_SEED_VARIANTS]
                dropped += len(t.spoken) - len(keep)
                t.spoken = keep
        self._compile()
        return dropped

    @property
    def revision(self) -> str:
        """A content hash. Stable across reloads and restarts, so an editor holding a stale
        list is refused no matter how the current list came to be (window, CLI, hand edit)."""
        import hashlib

        return hashlib.sha1(json.dumps(self.as_dicts(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]

    def save(self, path: Path = VOCAB_PATH) -> None:
        """Atomic and serialized: the window, the CLI and a hand edit may all write this file.
        A unique temp file per writer plus rename means a reader never sees a half-written
        list and two writers never trip over the same temp name."""
        with vocab_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".vocabulary-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"version": VOCAB_VERSION, "terms": [asdict(t) for t in self.terms]},
                                       indent=2, ensure_ascii=False) + "\n")
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    def as_dicts(self) -> list[dict]:
        return [asdict(t) for t in self.terms]

    # -- editing ----------------------------------------------------------------------
    def add(self, term: str, spoken: list[str] | None = None, note: str = "", fuzzy: bool | None = None) -> Term:
        """``fuzzy`` None = leave an existing term's setting alone (a new term starts off)."""
        term = term.strip()
        if not term:
            raise ValueError("empty term")
        spoken = [s.strip() for s in (spoken or []) if s.strip()]
        for t in self.terms:
            if t.term.lower() == term.lower():
                t.term = term
                t.spoken = list(dict.fromkeys(t.spoken + spoken))
                if note:
                    t.note = note
                if fuzzy is not None:
                    t.fuzzy = fuzzy
                self._compile()
                return t
        t = Term(term, spoken, note, fuzzy=bool(fuzzy))
        self.terms.append(t)
        self._compile()
        return t

    def remove(self, term: str) -> bool:
        before = len(self.terms)
        self.terms = [t for t in self.terms if t.term.lower() != term.strip().lower()]
        self._compile()
        return len(self.terms) != before

    def replace_all(self, terms: list[dict]) -> None:
        self.terms = [t for t in (Term.from_dict(d) for d in terms) if t.term]
        self._compile()

    # -- the three consumers ----------------------------------------------------------
    def _compile(self) -> None:
        rules: list[tuple[int, str, str]] = []
        for t in self.terms:
            variants = list(t.spoken)
            if _stylized(t.term):
                variants.append(t.term)
            for v in variants:
                if _fold(v) == _fold(t.term) and not _stylized(t.term):
                    continue                        # identical spelling; nothing to fix
                rx = _variant_rx(v)
                if not rx:
                    continue
                try:
                    re.compile(rx, re.IGNORECASE)
                except re.error as e:
                    log.warning("bad spoken variant %r for %s: %s", v, t.term, e)
                    continue
                rules.append((len(_fold(v)), rx, t.term))
        # Longest variant first: "ackme labs" must win over "ackme" at the same position. All the
        # variants go into ONE alternation and ONE pass, so a replacement can never be matched
        # again by a shorter rule ("acmelabs.io" → "AcmeLabs.io" was exactly that).
        rules.sort(key=lambda r: -r[0])
        self._rules = [(rx, term) for _, rx, term in rules]
        by_group = {f"v{i}": term for i, (_, term) in enumerate(self._rules)}
        rx = (re.compile("|".join(f"(?P<v{i}>{p})" for i, (p, _) in enumerate(self._rules)), re.IGNORECASE)
              if self._rules else None)
        # One attribute, assigned once: apply() on the model worker snapshots it, so a take is
        # never half-replaced with one revision and half with the next while the window or the
        # CLI edits the list on another thread.
        self._compiled = (rx, by_group)
        self._rx, self._by_group = rx, by_group

    def apply(self, text: str) -> str:
        """Deterministic spelling fixes (the listed variants), then sound-alike matching for
        capitalized words the speech model invented. Runs on the raw transcript and again on
        the LLM output."""
        rx, by_group = self._compiled
        if not text:
            return text
        if rx is not None:
            text = rx.sub(lambda m: by_group[m.lastgroup], text)
        return self.fuzzy(text)

    def add_learned(self, term: str, heard: str) -> Term:
        """A fix taught "heard → term". The spelling goes to the term's LEARNED list, which is
        applied through the same gates as sound-alike matching (a capitalized word inside a
        sentence, never a protected word) — a learned "Whisper → Wispr" can never rewrite the
        lowercase word "whisper"."""
        t = self.add(term)                       # fuzzy is left exactly as it was (never re-armed by a fix)
        heard = heard.strip()
        if heard and _fold(heard) not in {_fold(x) for x in t.learned} and _fold(heard) != _fold(t.term):
            t.learned.append(heard)
        self._compile()
        return t

    def _match_term(self, window: str) -> tuple[Term | None, bool]:
        """(term, learned) — the term this window most plausibly is, or None. A spelling the user
        taught matches exactly; sound-alike matching needs a clear best term (FUZZY_MARGIN over
        the runner-up)."""
        w = _fold(window)
        for t in self.terms:
            if t.term and any(_fold(x) == w for x in t.learned):
                return t, True
        scored = sorted(((similarity(window, t.term), t) for t in self.terms if t.term and t.fuzzy),
                        key=lambda p: -p[0])
        if not scored or scored[0][0] <= 0.0:
            return None, False
        if len(scored) > 1 and scored[1][0] > 0.0 and scored[0][0] - scored[1][0] < FUZZY_MARGIN:
            log.debug("fuzzy: %r is ambiguous between %s and %s — left alone", window, scored[0][1].term, scored[1][1].term)
            return None, False
        return scored[0][1], False

    def fuzzy(self, text: str) -> str:
        """Replace sentence-internal capitalized words (and short capitalized phrases) that are a
        learned spelling of a term, or sound like a term, with that term. Never a protected
        (ordinary) word, never across punctuation, never a bullet or a quote opener. Each hit
        is logged so it can be audited."""
        if not text or not self.protected:
            return text                          # no word list → fail closed
        candidates = [t for t in self.terms if t.term and (t.fuzzy or t.learned)]
        if not candidates:
            return text
        known = {_fold(t.term) for t in self.terms}
        for t in self.terms:
            known.update(_fold(s) for s in t.spoken)
        tokens = list(_TOKEN_RX.finditer(text))
        if not tokens:
            return text
        # Shortest window first: a one-word mishearing of a two-word term ("Wisprflow") must be
        # caught on its own token and never swallow the word after it. A longer
        # window only forms when every word in it is capitalized ("Whisper Flow", "Claude Code").
        lengths = sorted({len(t.term.split()) for t in candidates} | {1})
        out: list[str] = []
        pos = 0
        i = 0
        while i < len(tokens):
            replaced = False
            tok = tokens[i].group(0)
            if _name_like(tok, text[:tokens[i].start()]):
                for n in lengths:
                    if i + n > len(tokens):
                        continue
                    window = text[tokens[i].start(): tokens[i + n - 1].end()]
                    if (n > 1 and _WINDOW_BAD_RX.search(window)) or _fold(window) in known:
                        continue
                    if n > 1 and not all(tokens[i + k].group(0)[:1].isupper() for k in range(n)):
                        continue                 # "Wisprflow ayer": the second word is not part of a name
                    if "." in window and not any(_fold(window) == _fold(x) for tt in candidates for x in [tt.term] + tt.learned):
                        continue                 # "Supabase.com" is a host, not "Supabase" misheard
                    t, learned = self._match_term(window)
                    if t is None or (not learned and n > 1 and len(t.term.split()) != n):
                        continue                 # a learned spelling replaces whatever it spans; a sound-alike
                                                 # one-word window may stand for a multi-word term (Wisprflow)
                    if not learned and n == 1 and _is_plain_word(_fold(tok), self.protected):
                        continue                 # "Cloud", "Codecs", "Simon": a word or a name of its own
                    log.info("%s: %s → %s", "learned" if learned else "fuzzy", window, t.term)
                    out.append(text[pos: tokens[i].start()])
                    out.append(t.term)
                    pos = tokens[i + n - 1].end()
                    i += n
                    replaced = True
                    break
            if not replaced:
                i += 1
        out.append(text[pos:])
        return "".join(out)

    def whisper_terms(self) -> list[str]:
        """The spellings Whisper should be primed with (its prompt is the cheapest place to win)."""
        seen: dict[str, None] = {}
        for t in self.terms[:MAX_PROMPT_TERMS]:
            seen.setdefault(t.term, None)
        return list(seen)

    def glossary(self, max_variants: int = 2) -> str:
        """One compact line for the LLM system prompt (empty when there are no terms)."""
        if not self.terms:
            return ""
        parts = []
        for t in self.terms:
            hints = []
            if t.note:
                hints.append(t.note)
            if t.spoken:
                hints.append("heard as " + ", ".join(f"'{s}'" for s in t.spoken[:max_variants]))
            parts.append(t.term + (f" ({'; '.join(hints)})" if hints else ""))
        return "; ".join(parts)

    def __len__(self) -> int:
        return len(self.terms)
