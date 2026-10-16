"""Unit tests for dictdact.vocab's sound-alike matching: sound_key, sounds_like, and
Vocabulary.fuzzy/apply for terms that carry NO spoken variants at all (the "learn how I say
things" fuzzy path — distinct from the deterministic spoken-variant rules already covered in
test_vocab.py).

No ML models; pure regex/text. tests/conftest.py isolates DICTDACT_CONFIG_DIR/DATA_DIR.
"""
from dataclasses import asdict

import pytest

from dictdact.vocab import Term, Vocabulary, sound_key, sounds_like


# -- sound_key --------------------------------------------------------------------------

def test_sound_key_folds_hyrox_spellings_to_the_same_key():
    assert sound_key("Hirox") == sound_key("Hyrox") == sound_key("Hierox")


def test_sound_key_folds_manychat_spellings_to_the_same_key():
    assert sound_key("Manichat") == sound_key("ManyChat")


def test_sound_key_is_case_insensitive():
    assert sound_key("Whoop") == sound_key("WHOOP") == sound_key("woop")


def test_sound_key_folds_simon_and_simone_to_the_same_key():
    assert sound_key("Simon") == sound_key("Simone")


def test_sound_key_of_empty_string_is_empty():
    assert sound_key("") == ""


# -- sounds_like --------------------------------------------------------------------------

@pytest.mark.parametrize("candidate, term", [
    ("Hirox", "Hyrox"),
    ("Hierox", "Hyrox"),
    ("Superbase", "Supabase"),
    ("Manichat", "ManyChat"),
    ("Dictac", "DictDact"),
    ("Hicksfield", "Higgsfield"),
    ("Bercel", "Vercel"),
    ("Simon", "Simone"),
])
def test_sounds_like_true_for_known_mishearings(candidate, term):
    assert sounds_like(candidate, term) is True


@pytest.mark.parametrize("candidate, term", [
    ("Luna", "Simone"),       # unrelated sound
    ("super", "Supabase"),    # a real ordinary-word prefix, not a mishearing
    ("many", "ManyChat"),     # ditto
    ("Hyrox", "Hyrox"),       # identical -> not a "mishearing" of itself
    ("abc", "abcd"),          # shorter than FUZZY_MIN_CHARS (4)
])
def test_sounds_like_false_for_unrelated_identical_or_too_short(candidate, term):
    assert sounds_like(candidate, term) is False


# -- Vocabulary.fuzzy / apply — terms with no spoken variants at all -----------------------

def _fuzzy_vocab():
    terms = [Term("Hyrox", fuzzy=True), Term("Supabase", fuzzy=True), Term("ManyChat", fuzzy=True), Term("DictDact", fuzzy=True), Term("Higgsfield", fuzzy=True),
             Term("WHOOP", fuzzy=True), Term("Wispr Flow", fuzzy=True), Term("Simone", fuzzy=True), Term("Codex", fuzzy=True), Term("Claude", fuzzy=True), Term("Vercel", fuzzy=True)]
    return Vocabulary(terms)


def test_fuzzy_catches_several_mishearings_in_one_sentence():
    v = _fuzzy_vocab()
    out = v.apply("fuimos a Hirox y luego a Hierox con Superbase y Manichat")
    assert out == "fuimos a Hyrox y luego a Hyrox con Supabase y ManyChat"


def test_fuzzy_leaves_lowercase_words_and_sentence_initial_capitals_untouched():
    v = _fuzzy_vocab()
    text = "the audio codecs are fine. Codecs are fine. many chat messages, the super base"
    assert v.apply(text) == text


def test_fuzzy_matches_a_multiword_term():
    v = _fuzzy_vocab()
    assert v.apply("vamos con Whisper Flow") == "vamos con Wispr Flow"


def test_fuzzy_matches_all_caps_anywhere_in_the_sentence():
    v = _fuzzy_vocab()
    assert v.apply("WOOP. mi WOOP") == "WHOOP. mi WHOOP"


def test_fuzzy_only_fixes_the_word_that_actually_sounds_like_a_term():
    v = Vocabulary([Term("Hyrox", fuzzy=True), Term("Simone", fuzzy=True)])
    # "Luna" is not close to anything; "Simon" is a real name (in the system word list) — both stay;
    # "Hirox" is nobody's word and sounds like Hyrox — fixed.
    assert v.apply("talk to Luna tomorrow and to Simon after about Hirox") == \
        "talk to Luna tomorrow and to Simon after about Hyrox"


def test_fuzzy_can_be_disabled_per_term():
    v = Vocabulary([Term("Hyrox", fuzzy=False)])
    assert v.apply("fuimos a Hirox") == "fuimos a Hirox"


def test_protected_words_are_never_fuzzed():
    v = Vocabulary([Term("Claude", fuzzy=True)])
    v.protected = frozenset({"cloud"})
    assert v.apply("ask Cloud about it") == "ask Cloud about it"
    assert v.apply("ask Clod about it") == "ask Claude about it"


def test_a_correctly_spelled_term_is_left_alone():
    v = Vocabulary([Term("Hyrox", fuzzy=True)])
    assert v.apply("Hyrox is great") == "Hyrox is great"


def test_explicit_spoken_variant_still_runs_before_fuzzy():
    v = Vocabulary([Term("Acme Labs", ["ack me labs"])])
    assert v.apply("ack me labs") == "Acme Labs"


# -- Term.fuzzy round trip ------------------------------------------------------------------

def test_term_fuzzy_is_opt_in():
    assert Term.from_dict({"term": "X"}).fuzzy is False
    assert Term.from_dict({"term": "X", "fuzzy": True}).fuzzy is True
    v = Vocabulary([])
    assert v.add("Plain").fuzzy is False and v.add("Sound", fuzzy=True).fuzzy is True
    assert v.add("Sound").fuzzy is True                    # None leaves an existing setting alone
    assert v.add("Sound", fuzzy=False).fuzzy is False


def test_term_fuzzy_false_is_honored_from_dict():
    assert Term.from_dict({"term": "X", "fuzzy": False}).fuzzy is False


def test_asdict_keeps_the_fuzzy_flag():
    from dataclasses import asdict
    assert asdict(Term("X", fuzzy=True))["fuzzy"] is True
    assert asdict(Term("X"))["fuzzy"] is False
    assert Term.from_dict(asdict(Term("X", fuzzy=True))).fuzzy is True


# -- idempotency, seeds survive, punctuation, bullets, margin ---------------------------------

def test_apply_is_idempotent_and_dotted_terms_survive():
    from dictdact.vocab import DEFAULT_TERMS
    terms = list(DEFAULT_TERMS) + [Term("Acmetunes.me", ["acme tunes dot me", "acmetunesme"]),
                                   Term("Grupo Vela", fuzzy=True)]
    v = Vocabulary(terms)
    corpus = ["sube la cancion a Acmetunes.me hoy", "escucha Acmetunes.me hoy", "deploy en Vercel.app y Acmelabs.io",
              "fuimos a Hirox con Superbase y Manichat", "el Grupo. Velas subieron mucho", "la lista: Wispr, Flow y otros",
              "- Codecs revisados\n- Simona llamo", 'dijo: "Simona vino"', "usamos Clode para esto",
              "los nuevos Codecs estan bien", "hable con Simona y con Simoni"]
    for text in corpus:
        once = v.apply(text)
        assert v.apply(once) == once, text
    for t in terms:
        sentence = f"hoy usamos {t.term} otra vez"
        assert v.apply(sentence) == sentence, t.term


def test_multiword_windows_never_cross_punctuation_or_swallow_words():
    from dictdact.vocab import DEFAULT_TERMS
    v = Vocabulary(list(DEFAULT_TERMS) + [Term("Wispr Flow", fuzzy=True), Term("Grupo Vela", fuzzy=True)])
    assert v.apply("el Grupo. Velas subieron mucho") == "el Grupo. Velas subieron mucho"
    assert v.apply("hoy vimos: Whisper. Flow y otros") == "hoy vimos: Whisper. Flow y otros"
    # (a comma between the words of a listed multi-word term is still joined — "Dict, Dact" is a
    # real Whisper artifact and that is the explicit-variant engine, not sound-alike matching)


def test_bullets_and_quotes_start_a_sentence():
    v = Vocabulary([Term("Codex", fuzzy=True), Term("Simone", fuzzy=True)])
    v.protected = frozenset()                  # even with no word list…
    assert v.apply("- Codecs revisados\n- Simona llamo") == "- Codecs revisados\n- Simona llamo"
    assert v.apply('dijo: "Simona vino"') == 'dijo: "Simona vino"'
    assert v.apply("1) Codecs primero") == "1) Codecs primero"


def test_seeded_fuzzy_off_terms_and_acronyms_do_not_fuzz():
    from dictdact.vocab import DEFAULT_TERMS
    v = Vocabulary(list(DEFAULT_TERMS))
    for text in ("los nuevos Codecs", "el MLXS avanza", "usamos Chaterbox hoy", "usamos Chatterbocks hoy"):
        assert v.apply(text) == text, text


def test_ambiguous_candidate_between_two_terms_is_left_alone():
    v = Vocabulary([Term("Claude", fuzzy=True), Term("Codex", fuzzy=True)])
    v.protected = frozenset({"x"})            # a non-empty list: fuzzy is on
    assert v.apply("usamos Clode para esto") == "usamos Clode para esto"


def test_no_word_list_means_no_sound_alike_matching():
    v = Vocabulary([Term("Hyrox", fuzzy=True)])
    v.protected = frozenset()
    assert v.apply("fuimos a Hirox") == "fuimos a Hirox"


def test_learned_spellings_apply_only_where_a_name_can_stand():
    v = Vocabulary([])
    v.protected = frozenset({"whisper"})
    t = v.add_learned("Wispr", "Whisper")
    assert t.learned == ["Whisper"] and t.spoken == [] and t.fuzzy is False   # a fix never arms sound-alike matching
    assert v.apply("probamos Whisper para dictar") == "probamos Wispr para dictar"
    assert v.apply("el modelo whisper corre local") == "el modelo whisper corre local"     # lowercase: never
    assert v.apply("Whisper es rapido") == "Whisper es rapido"                             # sentence start: never


# -- hosts, URLs and learned spellings on seeds ------------------------------------------------

def test_hosts_and_urls_are_never_truncated_to_a_term():
    v = Vocabulary([Term("Supabase", fuzzy=True), Term("ManyChat", fuzzy=True), Term("Cloudflare", fuzzy=True)])
    for text in ("entra a Supabase.com hoy", "usa ManyChat.io ahora", "ve a Cloudflare.com/docs", "mira Supabase.com/docs"):
        assert v.apply(text) == text, text


def test_a_fix_never_rearms_sound_alike_matching_on_a_seed():
    from dictdact.vocab import DEFAULT_TERMS
    v = Vocabulary(list(DEFAULT_TERMS))
    v.add_learned("Chatterbox", "Chaterbox")
    assert next(t for t in v.terms if t.term == "Chatterbox").fuzzy is False
    assert v.apply("usamos Chatterbocks hoy") == "usamos Chatterbocks hoy"
    assert v.apply("usamos Chaterbox hoy") == "usamos Chatterbox hoy"     # the learned spelling itself does apply


# -- multi-word terms and one-word mishearings ------------------------------------------------

def test_a_one_word_mishearing_of_a_two_word_term_never_swallows_the_next_word():
    v = Vocabulary([Term("Wispr Flow", fuzzy=True), Term("Grupo Vela", fuzzy=True)])
    assert v.apply("abri Wisprflow ayer en el mac") == "abri Wispr Flow ayer en el mac"
    assert v.apply("hoy uso Wisprflow para dictar") == "hoy uso Wispr Flow para dictar"
    assert v.apply("vamos con Whisper Flow ayer") == "vamos con Wispr Flow ayer"
    assert v.apply("con Grupo Vela para todos") == "con Grupo Vela para todos"
    for text in ("abri Wisprflow ayer en el mac", "vamos con Whisper Flow ayer"):
        assert v.apply(v.apply(text)) == v.apply(text)


def test_a_learned_spelling_with_a_different_word_count_still_applies():
    v = Vocabulary([Term("Wispr Flow")])
    v.protected = frozenset({"x"})
    v.add_learned("Wispr Flow", "Wisprflow")
    assert v.apply("abri Wisprflow ayer") == "abri Wispr Flow ayer"
    v.add_learned("ManyChat", "Many Chat")
    assert v.apply("uso Many Chat hoy") == "uso ManyChat hoy"
