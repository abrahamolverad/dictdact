"""Learning how the speaker says things — from their own history and from their fixes.

Two sources, both local:
- history.jsonl: every take's raw transcript. Names Whisper is unsure about show up as several
  spellings of the same sound (Hyrox / Hirox / Hierox); clustering them by sound key surfaces
  the terms worth adding, with the evidence.
- a fix: the user edits the text of a dictation in the window. A word-level diff of what was
  pasted against what they wanted yields "heard X → write Y" pairs; name-like ones become
  dictionary entries on the spot and are logged to learned.jsonl.
"""
from __future__ import annotations

import collections
import difflib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR
from .vocab import Vocabulary, _fold, sound_key, sounds_like, system_words

log = logging.getLogger("dictdact.learn")

LEARNED_PATH = DATA_DIR / "learned.jsonl"

# Capitalized words that are not names: sentence openers Whisper capitalizes, days, months,
# languages, common es/en words that happen to be capitalized mid-sentence.
_STOP_CAPS = frozenset(_fold(w) for w in (
    "I I'm I'll I've I'd The And But So Also For Then This That Now Let What When How Yes No Okay Ok Please Make "
    "Just Can Not Because Which There Here Another One Two Three Sure Right Well Good Thank Thanks You Your Like "
    "They She He We It Its With From Those These Are Was Were Should Would Could Sometimes Whereas However Allow "
    "Share Post Plus Ultra Pro Business Computer Bank Free Day Days Week Month Year Today Tomorrow Yesterday Tonight "
    "Morning Night Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June "
    "July August September October November December English Spanish "
    "Bueno Entonces Para Pero Que Qué Como Cómo Esto Esta Este Está Estoy Hola Por Con Los Las Una Uno Del Muy "
    "Ahora Todo Todos Tengo Quiero Vamos Hay Creo Necesito Dame Haz Otra Otro Ese Esa Eso Mira Oye Perfecto "
    "Gracias Listo Igual Luego Después También Solo Nada Algo Cuando Donde Sin Puedes Puedo Ahorita Incluso "
    "Tienes Dime Porque Recuerda Agrega Soy Usa Vale Lunes Martes Miércoles Jueves Viernes Sábado Domingo Enero "
    "Febrero Marzo Abril Mayo Junio Julio Agosto Septiembre Octubre Noviembre Diciembre Español Inglés"
).split())

_CAP_TOKEN_RX = re.compile(r"\b([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ'\-]{2,})\b")
_SENTENCE_START_RX = re.compile(r"(?:^|[.!?¿¡\n]\s*)$")


@dataclass
class Suggestion:
    term: str                        # the most frequent spelling — the user may correct it
    spoken: list[str]                # the other spellings Whisper produced for the same sound
    count: int                       # how many takes carried any of them
    spellings: dict = field(default_factory=dict)   # spelling → count (evidence)
    strong: bool = False             # spellings differ beyond case/accents (Bebel/Bevel), not just WhatsApp/Whatsapp


def capitalized_names(text: str, common: set[str] | None = None, include_initial: bool = False) -> list[str]:
    """Capitalized words that look like names. In a transcript only the sentence-internal ones
    count (Whisper capitalizes every sentence opener); in screen text (``include_initial``)
    titles and list items start lines, so an opener is a name too unless it is a stop word.
    ``common`` = folded words the speaker uses in lowercase too (from their history)."""
    out = []
    for m in _CAP_TOKEN_RX.finditer(text):
        w = m.group(1)
        f = _fold(w)
        if f in _STOP_CAPS or (common and f in common):
            continue
        if _SENTENCE_START_RX.search(text[: m.start()]) and not (w.isupper() and len(w) >= 3):
            if not include_initial:
                continue                         # a sentence opener; all-caps counts anywhere
            # screen text: a short line ("Hyrox", "Simone · Level 12") is a title or a list item
            # and its first word is a name; a long line is a sentence and its opener is not.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line = text[line_start: line_end if line_end != -1 else len(text)]
            if len(line.split()) > 4:
                continue
        out.append(w)
    return out


def common_words(raws: list[str]) -> set[str]:
    """Words the speaker also uses in lowercase: never names. Data, not a list."""
    lower: collections.Counter = collections.Counter()
    upper: collections.Counter = collections.Counter()
    for raw in raws:
        for m in re.finditer(r"[A-Za-zÁÉÍÓÚÑáéíóúñ']{3,}", raw):
            w = m.group(0)
            (upper if w[0].isupper() else lower)[_fold(w)] += 1
    return {w for w, n in lower.items() if n >= upper.get(w, 0)}


def protected_words(raws: list[str]) -> frozenset[str]:
    """What sound-alike matching must leave alone: the speaker's own ordinary words (from their
    history) plus the system's English word list."""
    return frozenset(common_words(raws)) | system_words()


def suggest(raws: list[str], vocab: Vocabulary, limit: int = 30, min_count: int = 3, min_alt: int = 2,
            texts: list[str] | None = None) -> list[Suggestion]:
    """Cluster the names in the speaker's takes by sound; a cluster with several spellings is
    a name Whisper is unsure about — exactly what the dictionary is for. Clusters that sound
    like an existing term are skipped (fuzzy matching already covers them). ``min_alt``: the
    other spellings must show up in at least this many takes — one stray "Italia" next to
    four "Italy" is noise, three "Hirox" next to six "Hyrox" is a pattern."""
    common = common_words(raws)
    known_keys = {sound_key(t.term) for t in vocab.terms if t.term}
    known_folded = {_fold(t.term) for t in vocab.terms} | {_fold(s) for t in vocab.terms for s in t.spoken}
    clusters: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    takes: dict[str, set[int]] = collections.defaultdict(set)
    takes_by_spelling: dict[str, dict[str, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for i, raw in enumerate(raws):
        for w in capitalized_names(raw, common):
            k = sound_key(w)
            if not k or k in known_keys or _fold(w) in known_folded:
                continue
            clusters[k][w] += 1
            takes[k].add(i)
            takes_by_spelling[k][w].add(i)
    # Two clusters whose top spellings sound alike are one name (Hicksfield / Higgsfield share no
    # exact sound key but do sound alike): merge them.
    for k in [k for k, ts in takes.items() if len(ts) < 2]:
        clusters.pop(k, None); takes.pop(k, None); takes_by_spelling.pop(k, None)   # singletons are noise
    keys = sorted(clusters, key=lambda k: -sum(clusters[k].values()))
    merged_into: dict[str, str] = {}
    for a_i, a in enumerate(keys):
        if a in merged_into:
            continue
        top_a = clusters[a].most_common(1)[0][0]
        for b in keys[a_i + 1:]:
            if b in merged_into:
                continue
            top_b = clusters[b].most_common(1)[0][0]
            if min(len(top_a), len(top_b)) < 6:
                continue                       # short names collide by accident (Anna / Banana)
            if sounds_like(top_b, top_a) or sounds_like(top_a, top_b):
                clusters[a].update(clusters[b])
                takes[a] |= takes[b]
                for w, ts in takes_by_spelling[b].items():
                    takes_by_spelling[a][w] |= ts
                merged_into[b] = a
    for b in merged_into:
        clusters.pop(b, None)
    # The formatter (a real LLM) usually knows the right spelling of a product name even when
    # Whisper does not: prefer the spelling that dominates the FORMATTED texts as the canonical.
    formatted: collections.Counter = collections.Counter()
    for t in texts or []:
        for w in capitalized_names(t, common):
            formatted[w] += 1
    out: list[Suggestion] = []
    for k, spell in clusters.items():
        if len(takes[k]) < min_count or len(spell) < 2:
            continue
        ranked = spell.most_common()
        stylized = lambda w: w != w.capitalize() and w != w.lower()      # ManyChat, WHOOP: a written brand
        term = max((w for w, _ in ranked), key=lambda w: (formatted.get(w, 0), stylized(w), spell[w]))
        base = _fold(term).rstrip("s'-")
        others = [w for w, _ in ranked if w != term
                  and not (_fold(w) != _fold(term) and _fold(w).rstrip("s'-") == base)    # "APIs", "GPT-", "Amelie's"
                  and (sounds_like(w, term) or sound_key(w) == sound_key(term) or _fold(w) == _fold(term))]
        if not others:
            continue
        alt_takes = set().union(*(takes_by_spelling[k][w] for w in others))
        if len(alt_takes) < min_alt:
            continue
        strong = any(_fold(w) != _fold(term) for w in others)
        out.append(Suggestion(term=term, spoken=others, count=len(takes[k]), spellings=dict(ranked), strong=strong))
    out.sort(key=lambda s: (not s.strong, -s.count))
    return out[:limit]


# --- learning from a fix -------------------------------------------------------------------
_WORD_RX = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-.]*")


def _name_like(phrase: str) -> bool:
    words = phrase.split()
    return bool(words) and (any(w[:1].isupper() for w in words) or bool(re.search(r"[0-9.\-]", phrase)))


def learn_from_fix(old: str, new: str, max_words: int = 3) -> list[tuple[str, str]]:
    """Word-level diff of the pasted text against the user's fix → (heard, wanted) pairs that
    are spelling corrections of a name: both sides carry letters, the wanted side looks like a
    name, the heard side is not an ordinary word or a real name in its own right, and the two
    sides sound alike. "John" → "Daniel", "2.5" → "2.6" and "Erik" → "Eric" teach nothing."""
    a = [m.group(0) for m in _WORD_RX.finditer(old)]
    b = [m.group(0) for m in _WORD_RX.finditer(new)]
    words = system_words()
    pairs: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [_fold(w) for w in a], [_fold(w) for w in b],
                                                     autojunk=False).get_opcodes():
        if op != "replace" or i2 - i1 > max_words or j2 - j1 > max_words:
            continue
        heard, wanted = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if re.search(r"\d", heard + wanted) or not (re.search(r"[A-Za-zÀ-ÿ]", heard) and re.search(r"[A-Za-zÀ-ÿ]", wanted)):
            continue                                  # numbers, versions, models (R2 → R3): never a spelling
        if not _name_like(wanted) or _fold(heard) == _fold(wanted):
            continue
        if any(_fold(w) in words for w in heard.split()):
            continue                                  # "Whisper", "Erik": a word / a name of its own
        la, lb = re.sub(r"[^a-z]", "", _fold(heard)), re.sub(r"[^a-z]", "", _fold(wanted))
        if la != lb:                                  # same letters = a stylization fix (Acmeo → Acme-O): learn it
            common = len(os.path.commonprefix([la, lb]))
            if min(len(la), len(lb)) >= 5 and max(len(la), len(lb)) - common <= 2:
                continue                              # Alejandra → Alejandro, Presupuesto → Presupuestos, Amelie → Amelia:
                                                      # a different ending is a different word, not a mishearing
        if sounds_like(heard, wanted) or (sound_key(heard) and sound_key(heard) == sound_key(wanted)):
            pairs.append((heard, wanted))
    return pairs


def record(pairs: list[tuple[str, str]], block_id: str | None = None, path: Path = LEARNED_PATH) -> None:
    if not pairs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for heard, wanted in pairs:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "heard": heard, "wanted": wanted,
                                "block": block_id}, ensure_ascii=False) + "\n")


def apply_learned(vocab: Vocabulary, pairs: list[tuple[str, str]]) -> list[dict]:
    """Add each (heard, wanted) pair to the dictionary: ``wanted`` becomes (or extends) a term,
    ``heard`` one of its LEARNED spellings — applied only where a name can stand, never to a
    lowercase ordinary word. Returns what changed, for the toast."""
    changed = []
    for heard, wanted in pairs:
        t = vocab.add_learned(wanted, heard)
        changed.append({"heard": heard, "term": t.term})
    return changed


def suggestions_as_dicts(items: list[Suggestion]) -> list[dict]:
    return [asdict(s) for s in items]
