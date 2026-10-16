"""Unit tests for dictdact.learn: clustering mis-heard names out of the speaker's own history
(``suggest``), and learning corrections from a fix the user makes in the window
(``learn_from_fix`` / ``record`` / ``apply_learned``).

No ML models; pure text/regex plus small jsonl file I/O. tests/conftest.py isolates
DICTDACT_CONFIG_DIR/DATA_DIR — every test that touches disk here passes its own tmp_path.
"""
import json

from dictdact.learn import apply_learned, capitalized_names, common_words, learn_from_fix, record, suggest
from dictdact.vocab import Term, Vocabulary


# -- capitalized_names --------------------------------------------------------------------

def test_capitalized_names_finds_mid_sentence_names_not_sentence_openers():
    out = capitalized_names("Hoy vimos a Hyrox y a ManyChat. Luego Simone llegó")
    assert out == ["Hyrox", "ManyChat", "Simone"]
    assert "Hoy" not in out
    assert "Luego" not in out


def test_capitalized_names_all_caps_counts_even_at_sentence_start():
    assert "MLX" in capitalized_names("MLX is fine")


def test_capitalized_names_excludes_words_marked_common():
    out = capitalized_names("Hoy vimos a Hyrox y a ManyChat. Luego Simone llegó", common={"hyrox"})
    assert "Hyrox" not in out
    assert out == ["ManyChat", "Simone"]


# -- common_words -------------------------------------------------------------------------

def test_common_words_includes_a_word_also_seen_lowercase():
    words = common_words(["el cloud es grande", "usa Cloud hoy", "cloud cloud"])
    assert "cloud" in words


def test_common_words_excludes_a_word_only_ever_capitalized():
    words = common_words(["Hyrox es genial", "Hyrox de nuevo"])
    assert "hyrox" not in words


# -- suggest --------------------------------------------------------------------------------
# Synthetic take history: Hyrox (x3) is strictly the most frequent spelling of its sound,
# ahead of Hirox (x2) and a mid-sentence Hierox (x1); Bevel (x3) similarly dominates Bebel (x1);
# WhatsApp/Whatsapp (1 each, case-only difference) is a weak, low-count cluster.
RAWS = [
    "fuimos a Hyrox hoy",
    "fuimos a Hyrox de nuevo",
    "fuimos a Hyrox otra vez",
    "fuimos a Hirox ayer",
    "fuimos a Hirox el sabado",
    "en medio de la frase decimos Hierox y seguimos",
    "hicimos Bevel esta semana",
    "hicimos Bevel de nuevo",
    "hicimos Bevel otra vez",
    "hicimos Bebel el otro dia",
    "usamos WhatsApp para avisar",
    "probamos Whatsapp de nuevo",
]


def _by_term(suggestions, term):
    return next(s for s in suggestions if s.term == term)


# NOTE: suggest() also takes a ``min_alt`` parameter (default 2): the *alternate* spelling(s)
# of a cluster must show up in at least that many distinct takes, independent of ``min_count``.
# "Bebel" and "Whatsapp" above each occur in exactly one take, so under the real defaults they
# are filtered out by min_alt regardless of min_count (verified by running suggest(RAWS,
# Vocabulary([])) with min_count=2: the Bevel/WhatsApp clusters still do not appear). The tests
# below that need those clusters pass min_alt=1 explicitly, to exercise the min_count-based
# demotion/inclusion behavior in isolation from that separate min_alt gate.

def test_suggest_clusters_hyrox_mishearings_into_a_strong_suggestion():
    out = suggest(RAWS, Vocabulary([]))
    hyrox = _by_term(out, "Hyrox")
    assert set(hyrox.spoken) == {"Hirox", "Hierox"}
    assert hyrox.count >= 3
    assert hyrox.strong is True


def test_suggest_default_min_count_drops_the_whatsapp_cluster():
    out = suggest(RAWS, Vocabulary([]))
    assert not any(s.term.lower() == "whatsapp" for s in out)


def test_suggest_min_count_2_surfaces_whatsapp_as_a_weak_suggestion():
    out = suggest(RAWS, Vocabulary([]), min_count=2, min_alt=1)
    whatsapp = _by_term(out, "WhatsApp")
    assert whatsapp.count == 2
    assert whatsapp.strong is False


def test_suggest_orders_strong_suggestions_before_weak_ones():
    out = suggest(RAWS, Vocabulary([]), min_count=2, min_alt=1)
    flags = [s.strong for s in out]
    assert flags == sorted(flags, reverse=True)      # True (strong) sorts before False (weak)


def test_suggest_skips_a_cluster_that_already_matches_a_known_term():
    out = suggest(RAWS, Vocabulary([Term("Hyrox")]), min_alt=1)
    assert not any(s.term == "Hyrox" for s in out)
    assert any(s.term == "Bevel" for s in out)       # unrelated clusters are unaffected


# -- learn_from_fix ---------------------------------------------------------------------------

def test_learn_from_fix_learns_name_spelling_fixes_and_skips_a_content_change():
    # "superbase" is a real (chemistry) word in the system list: it is not learned; Manichat is
    assert learn_from_fix("fuimos a Hirox con Manichat y John", "fuimos a Hyrox con ManyChat y Daniel") == \
        [("Hirox", "Hyrox"), ("Manichat", "ManyChat")]


def test_learn_from_fix_pure_insertion_teaches_nothing():
    assert learn_from_fix("el reporte de ventas", "el reporte mensual de ventas") == []


def test_learn_from_fix_identical_texts_teach_nothing():
    assert learn_from_fix("same text here", "same text here") == []


def test_learn_from_fix_stylization_correction():
    # "cloud" is an ordinary English word: never learned as a spelling of Claude
    assert learn_from_fix("open Acmeo and cloud code", "open Acme-O and Claude Code") == [("Acmeo", "Acme-O")]


def test_record_with_no_pairs_creates_nothing(tmp_path):
    path = tmp_path / "learned.jsonl"
    record([], "blk", path=path)
    assert not path.exists()


# -- apply_learned --------------------------------------------------------------------------

def test_apply_learned_creates_a_term_with_the_heard_word_as_learned():
    v = Vocabulary([])
    out = apply_learned(v, [("Hirox", "Hyrox")])
    assert out == [{"heard": "Hirox", "term": "Hyrox"}]
    assert v.as_dicts() == [{"term": "Hyrox", "spoken": [], "note": "", "fuzzy": False, "learned": ["Hirox"]}]


def test_apply_learned_merges_a_second_spelling_into_the_same_term():
    v = Vocabulary([])
    apply_learned(v, [("Hirox", "Hyrox")])
    apply_learned(v, [("Hierox", "Hyrox")])
    assert len(v.terms) == 1 and v.terms[0].learned == ["Hirox", "Hierox"]


# -- what a fix must NOT teach ----------------------------------------------------------------

def test_learn_from_fix_ignores_numbers_words_and_real_names():
    assert learn_from_fix("la version 2.5 del modelo", "la version 2.6 del modelo") == []
    assert learn_from_fix("cuestan 100 pesos", "cuestan 1000 pesos") == []
    assert learn_from_fix("son las 9:30", "son las 9:45") == []
    assert learn_from_fix("probamos Whisper para dictar", "probamos Wispr para dictar") == []   # a real word
    assert learn_from_fix("send it to John", "send it to Daniel") == []
    assert learn_from_fix("fuimos a Hirox", "fuimos a Hyrox") == [("Hirox", "Hyrox")]


def test_suggest_prefers_the_formatted_spelling_and_merges_soundalike_clusters():
    raws = ["hoy fuimos a Hirox", "otra vez a Hirox", "y luego a Hirox", "vamos a Hyrox", "de nuevo Hyrox",
            "revisa Hicksfield hoy", "otra de Hicksfield", "y mas Hicksfield", "usa Higgsfield", "otra Higgsfield"]
    texts = ["hoy fuimos a Hyrox", "otra vez a Hyrox", "y luego a Hyrox", "vamos a Hyrox", "de nuevo Hyrox",
             "revisa Higgsfield hoy", "otra de Higgsfield", "y mas Higgsfield", "usa Higgsfield", "otra Higgsfield"]
    out = {s.term: s for s in suggest(raws, Vocabulary([]), texts=texts)}
    assert "Hyrox" in out and "Hirox" in out["Hyrox"].spoken          # the formatter's spelling wins
    assert "Higgsfield" in out and "Hicksfield" in out["Higgsfield"].spoken   # different sound keys, merged


def test_suggest_ignores_plural_and_dash_only_spellings_and_accidental_merges():
    raws = ["usamos la API hoy", "las APIs de pago", "otra API mas", "y las APIs otra vez",
            "mi amiga Anna vino", "otra vez Anna", "vimos a Anna", "comimos Banana ayer", "otra Banana", "mas Banana aqui"]
    out = {s.term: s for s in suggest(raws, Vocabulary([]), min_alt=1)}
    assert "API" not in out                       # "APIs" is a plural, not a spelling
    assert not any("Banana" in s.spoken for s in out.values())


def test_learn_from_fix_rejects_alphanumerics_and_ending_only_changes():
    assert learn_from_fix("las imagenes viven en R2", "las imagenes viven en R3") == []
    assert learn_from_fix("usa GPT-4 hoy", "usa GPT-5 hoy") == []
    assert learn_from_fix("habla con Alejandra", "habla con Alejandro") == []
    assert learn_from_fix("el Presupuesto final", "el Presupuestos final") == []
    assert learn_from_fix("vamos a Hierox", "vamos a Hyrox") == [("Hierox", "Hyrox")]
