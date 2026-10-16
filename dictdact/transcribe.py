"""Speech-to-text engines (all local, Apple-silicon MLX)."""
from __future__ import annotations

import logging
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import TARGET_SR, save_wav

log = logging.getLogger("dictdact.stt")

# Whisper's prompt is where language and spelling are cheapest to steer. Each prompt is
# dominant in ONE language: a mixed prompt used to make the first words of an English take
# come out Spanish ("In general" → "En general") because both readings were plausible.
#
# Auto mode keeps ONE code-switched clause so Spanish inside an English take (and vice versa)
# is still transcribed as spoken instead of translated. A locked language gets a monolingual
# prompt on top of the forced language token.
PROMPTS = {
    "es": "Hola, ¿cómo va todo? Vamos a revisar el plan para mañana, and then we check the onboarding, "
          "y después mandamos el update al equipo.",
    "en": "Hi, how is everything? Let's review the plan for tomorrow, y después vemos lo del onboarding, "
          "and then we send the update to the team.",
}
LOCKED_PROMPTS = {
    "es": "Hola, ¿cómo va todo? Vamos a revisar el plan para mañana y después mandamos el update al equipo.",
    "en": "Hi, how is everything? Let's review the plan for tomorrow and then send the update to the team.",
}
# Kept for the prompt-echo filter and for older tests; no longer sent to Whisper.
BILINGUAL_PROMPT = (
    "Hola, ¿cómo estás? I'm good, gracias. Vamos a revisar the plan para mañana, "
    "y después we can grab lunch."
)
SUPPORTED = ("es", "en")


def build_prompt(language: str | None, locked: bool, terms: list[str] | None = None) -> str:
    """The initial prompt for one take: the language's priming sentence plus the custom
    vocabulary as a tail (Whisper attends most to the end of the prompt, and it keeps only
    the last 223 tokens — the sentence goes first so the terms are what survives a long list)."""
    lang = language if language in SUPPORTED else "en"
    base = (LOCKED_PROMPTS if locked else PROMPTS)[lang]
    if terms:
        base += " " + ", ".join(terms) + "."
    return base


# Phrases Whisper produces on silence / noise. If the whole output is one of these, drop it.
_HALLUCINATIONS = [
    r"subt[ií]tulos? (realizados?|hechos?|por|creados?) .*",
    r".*amara\.org.*",
    r"thank(s| you)( for watching| so much)?\.?",
    r"gracias por (ver|mirar|su atenci[oó]n).*",
    r"(please )?(like|subscribe|suscr[ií]be).*",
    r"¡?suscr[ií]bete.*",
    r"you\.?",
    r"\.{2,}",
    r"(bye|adi[oó]s)[.!]?",
    r"www\..*",
]
_HALLUCINATION_RE = re.compile("^(" + "|".join(_HALLUCINATIONS) + ")$", re.IGNORECASE)

# A decoder loop: the same 1–8-word phrase, SIX or more times in a row ("no, no, no, no, no,
# no, no…", "I think, I think, I think, I think, I think, I think"). Emphatic speech repeats a
# word up to five times ("No, no, no, no, no voy a aceptar eso") and must survive;
# a single word therefore keeps three copies so the emphasis reads naturally.
LOOP_MIN_COPIES = 6
SENTENCE_LOOP_MIN = 4
_PHRASE_LOOP_RX = re.compile(
    r"(?i)\b((?:[^\s,.!?¿¡]+)(?:\s+[^\s,.!?¿¡]+){0,7}?)((?:([,.!?]*\s+)(\1)\b){%d,})" % (LOOP_MIN_COPIES - 1))
_SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")


def _norm_sentence(s: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()).strip()


def _collapse_phrase(m: re.Match) -> str:
    phrase, sep, repeat = m.group(1), m.group(3), m.group(4)
    if " " not in phrase.strip():
        return phrase + sep + repeat + sep + repeat     # "No, no, no" — emphasis kept, the loop gone
    return phrase


def collapse_repeats(text: str) -> str:
    """Collapse decoder repetition loops. Two passes: whole sentences repeated
    SENTENCE_LOOP_MIN or more times in a row ("Y luego, ¿cómo estás?" ×40 → one), then a
    phrase repeated LOOP_MIN_COPIES or more times inside a sentence (a single word keeps
    three copies, a longer phrase one)."""
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RX.split(text.strip())
    out: list[str] = []
    i = 0
    while i < len(sentences):
        j = i
        while j + 1 < len(sentences) and _norm_sentence(sentences[j + 1]) == _norm_sentence(sentences[i]):
            j += 1
        out.append(sentences[i])
        i = j + 1 if j - i + 1 >= SENTENCE_LOOP_MIN else i + 1
    text = " ".join(out)
    return _PHRASE_LOOP_RX.sub(_collapse_phrase, text)


@dataclass
class Transcript:
    text: str
    language: str | None = None
    engine: str = ""
    elapsed_s: float = 0.0
    segments: list[dict] = field(default_factory=list)


def _clean(text: str, prompt: str | None = None, terms_tail: str | None = None) -> str:
    text = text.strip()
    # Whisper occasionally echoes the initial prompt on near-silent audio — including the
    # vocabulary tail ("Acme Labs, ACML, Hyrox."). An echo is two or more prompt
    # sentences, or the vocabulary list itself; ONE matching sentence is a real dictation —
    # "Hola, ¿cómo va todo?" is something people actually say.
    sentences: list[str] = []
    for source in (prompt or "", BILINGUAL_PROMPT):
        sentences += [s.strip() for s in re.split(r"(?<=[.?!])\s+", source) if len(s.strip()) > 12]
    hits = [s for s in sentences if s.lower() in text.lower()]
    tail = (terms_tail or "").strip().lower()
    if len(hits) >= 2 or (tail and any(h.lower() == tail for h in hits)):
        for sentence in hits:
            text = re.sub(re.escape(sentence), "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text)
    collapsed = collapse_repeats(text)
    if collapsed != text:
        log.warning("repetition loop collapsed: %d → %d chars (%r)", len(text), len(collapsed), text[:80])
        text = collapsed
    if _HALLUCINATION_RE.match(text):
        log.info("dropped hallucination: %r", text)
        return ""
    return text


class WhisperEngine:
    name = "whisper"

    def __init__(self, model_repo: str, language: str | None = None, terms: list[str] | None = None):
        self.model_repo = model_repo
        self.language = language if language in SUPPORTED else None
        self.terms: list[str] = list(terms or [])
        import mlx_whisper  # noqa: F401  (import cost paid once, at startup)

        self._mod = mlx_whisper

    def _model(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        return ModelHolder.get_model(self.model_repo, mx.float16)

    def warmup(self) -> None:
        t0 = time.time()
        self._model()
        self.transcribe(np.zeros(TARGET_SR // 2, dtype=np.float32))
        log.info("whisper %s loaded in %.1fs", self.model_repo, time.time() - t0)

    def touch(self) -> None:
        """One tiny decode so every weight page stays resident (keep-warm; see pipeline)."""
        self.transcribe(np.zeros(TARGET_SR // 4, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, extra_terms: list[str] | None = None) -> Transcript:
        # The single-window shortcut below lost an 18.8 s dictation once ("Epa.") — Whisper
        # without timestamp tokens can stop early. Always use the seeking transcriber.
        return self._transcribe_long(audio, extra_terms)

    # -- language ----------------------------------------------------------------------
    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """One encoder pass over the first 30 s → the more likely of es/en and its probability.
        Whisper's 99-language argmax has produced 'tl', 'id' and 'pt' for this speaker; the
        app only ever wants Spanish or English, so the choice is made between those two."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

        model = self._model()
        mel = log_mel_spectrogram(audio[: TARGET_SR * 30], n_mels=model.dims.n_mels)
        seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(seg)
        if isinstance(probs, list):
            probs = probs[0] if probs else {}
        es, en = float(probs.get("es", 0.0)), float(probs.get("en", 0.0))
        lang = "es" if es >= en else "en"
        top = max(probs, key=probs.get) if probs else lang
        if top not in SUPPORTED:
            log.debug("whisper argmax language %s (%.2f) — choosing %s between es/en", top, probs[top], lang)
        return lang, max(es, en)

    DICT_SLOTS, CONTEXT_SLOTS = 32, 8     # of the ~40 terms the 223-token prompt window can carry

    def _all_terms(self, extra_terms: list[str] | None = None) -> list[str]:
        """The dictionary plus this take's context names (names visible in the active window),
        de-duplicated, each with its own reserved slice so a big dictionary can never crowd the
        context out, and capped so the sentence survives Whisper's prompt window."""
        seen: dict[str, None] = {}
        extras = [t for t in (extra_terms or []) if t][: self.CONTEXT_SLOTS]
        dict_slots = self.DICT_SLOTS + (self.CONTEXT_SLOTS - len(extras))    # unused context slots
        for t in list(self.terms)[:dict_slots]:
            if t and t.lower() not in {k.lower() for k in seen}:
                seen[t] = None
        added = 0
        for t in extra_terms or []:
            if added >= self.CONTEXT_SLOTS:
                break
            if t and t.lower() not in {k.lower() for k in seen}:
                seen[t] = None
                added += 1
        if len(self.terms) > dict_slots:
            log.debug("prompt tail: %d of %d dictionary terms fit", dict_slots, len(self.terms))
        return list(seen)

    def _prompt(self, language: str, locked: bool, extra_terms: list[str] | None = None) -> str:
        return build_prompt(language, locked, self._all_terms(extra_terms))

    def _terms_tail(self, extra_terms: list[str] | None = None) -> str | None:
        terms = self._all_terms(extra_terms)
        return (", ".join(terms) + ".") if terms else None

    def _transcribe_single_window(self, audio: np.ndarray) -> Transcript:
        """Dictation is almost always < 30 s = one Whisper window. Decode it directly:
        one encoder pass (language detection reuses it), no timestamp tokens, no extra
        GPU syncs — several times faster than the generic transcribe() under GPU load."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions, decode

        t0 = time.time()
        model = self._model()
        locked = self.language is not None
        language = self.language or self.detect_language(audio)[0]
        prompt = self._prompt(language, locked)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        opts = DecodingOptions(task="transcribe", language=language, temperature=0.0,
                               prompt=prompt, without_timestamps=True, fp16=True)
        res = decode(model, mel, opts)
        text = res.text
        if res.no_speech_prob > 0.85 and res.avg_logprob < -1.2:
            log.info("whisper judged no speech (p=%.2f, logprob=%.2f): %r", res.no_speech_prob, res.avg_logprob, text)
            text = ""
        elif res.compression_ratio > 2.4:
            log.info("repetitive output (ratio %.1f); retrying with the full decoder", res.compression_ratio)
            return self._transcribe_long(audio)
        return Transcript(
            text=_clean(text, prompt, self._terms_tail()),
            language=res.language,
            engine=f"whisper:{self.model_repo.split('/')[-1]}",
            elapsed_s=time.time() - t0,
            segments=[{"start": 0.0, "end": len(audio) / TARGET_SR, "text": text}],
        )

    def _transcribe_long(self, audio: np.ndarray, extra_terms: list[str] | None = None) -> Transcript:
        t0 = time.time()
        locked = self.language is not None
        if locked:
            language = self.language
        else:
            language, p = self.detect_language(audio)
            log.debug("language %s (p=%.2f)", language, p)
        prompt = self._prompt(language, locked, extra_terms)
        result = self._mod.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            language=language,
            initial_prompt=prompt,
            condition_on_previous_text=False,   # stops run-away repetition loops
            temperature=0.0,                    # greedy; sampling fallbacks only invent words on quiet audio
            no_speech_threshold=None,           # the user held the key and spoke: always try (whispers!)
            logprob_threshold=None,
            compression_ratio_threshold=2.4,
            fp16=True,
            verbose=None,
        )
        raw = result.get("text", "")
        text = _clean(raw, prompt, self._terms_tail(extra_terms))
        if raw.strip() and not text:
            log.info("whisper output dropped by the hallucination filter: %r", raw.strip()[:120])
        return Transcript(
            text=text,
            language=result.get("language") or language,
            engine=f"whisper:{self.model_repo.split('/')[-1]}",
            elapsed_s=time.time() - t0,
            segments=[{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result.get("segments", [])],
        )


class ParakeetEngine:
    """NVIDIA Parakeet TDT v3 via parakeet-mlx: no language token, so Spanish/English
    code-switching is decoded as-is. Very fast; punctuation is slightly weaker than Whisper.
    No prompt either — the vocabulary is applied by the formatter's replacements only."""

    name = "parakeet"

    def __init__(self, model_repo: str, language: str | None = None, terms: list[str] | None = None):
        self.model_repo = model_repo
        self.language = language
        self.terms: list[str] = list(terms or [])
        from parakeet_mlx import from_pretrained

        t0 = time.time()
        self._model = from_pretrained(model_repo)
        log.info("parakeet %s loaded in %.1fs", model_repo, time.time() - t0)

    def warmup(self) -> None:
        self.transcribe(np.zeros(TARGET_SR // 2, dtype=np.float32))

    def touch(self) -> None:
        """One tiny decode so every weight page stays resident (keep-warm; see pipeline)."""
        self.transcribe(np.zeros(TARGET_SR // 4, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, extra_terms: list[str] | None = None) -> Transcript:
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            save_wav(Path(tmp.name), audio)
            result = self._model.transcribe(tmp.name, chunk_duration=120.0)
        text = _clean(result.text)
        return Transcript(
            text=text,
            language=None,
            engine=f"parakeet:{self.model_repo.split('/')[-1]}",
            elapsed_s=time.time() - t0,
            segments=[{"start": s.start, "end": s.end, "text": s.text} for s in getattr(result, "sentences", [])],
        )


def make_engine(cfg, terms: list[str] | None = None) -> WhisperEngine | ParakeetEngine:
    if cfg.engine == "parakeet":
        return ParakeetEngine(cfg.parakeet_model, cfg.language, terms)
    if cfg.engine == "whisper":
        return WhisperEngine(cfg.whisper_model, cfg.language, terms)
    raise SystemExit(f"unknown engine {cfg.engine!r} (use 'whisper' or 'parakeet')")
