"""Unit tests for dictdact.vocab — the custom-spelling vocabulary.

No ML models involved; Vocabulary is pure regex/text. `tests/conftest.py` already isolates
DICTDACT_CONFIG_DIR / DICTDACT_DATA_DIR to a temp dir.
"""
import json

import pytest

from dictdact.vocab import MAX_PROMPT_TERMS, DEFAULT_TERMS, Term, Vocabulary

# Read-only across every test below: .apply() never mutates the underlying Term objects.
DEFAULT_VOCAB = Vocabulary(list(DEFAULT_TERMS))

# A fictional company and its acronym, the way a user would add them to the dictionary.
ACME_TERMS = [Term("Acme Labs", ["acne labs", "ack me labs", "ackme labs"], "a fictional company", fuzzy=False),
              Term("ACML", ["A C M L", "AC ML", "ACM L"], fuzzy=False)]


# -- typical Whisper mishearings --------------------------------------------------------

def test_user_terms_fix_typical_mishearings():
    v = Vocabulary(list(DEFAULT_TERMS) + ACME_TERMS)
    assert v.apply("¿Es Acne Labs? Sí, Ack Me Labs es mi compañía") == "¿Es Acme Labs? Sí, Acme Labs es mi compañía"
    assert v.apply("cuando digo Ack Me Labs, A-C-M-L, es mi empresa") == "cuando digo Acme Labs, ACML, es mi empresa"
    assert v.apply("para ackme-labs, pues") == "para Acme Labs, pues"
    assert v.apply("the AC ML.") == "the ACML."


def test_default_terms_fix_typical_mishearings():
    v = DEFAULT_VOCAB
    assert v.apply("Mi aplicación de Dict, Dact.") == "Mi aplicación de DictDact."
    assert v.apply("al nivel de Whisper Flow con WhisperFo") == "al nivel de Wispr Flow con Wispr Flow"
    assert v.apply("Superbase y Manichat") == "Supabase y ManyChat"
    assert v.apply("Hirox y Hierox") == "Hyrox y Hyrox"
    assert v.apply("my whoop data") == "my WHOOP data"


# -- spelled-out acronyms ---------------------------------------------------------------

def test_spelled_out_acronyms_all_normalize():
    v = DEFAULT_VOCAB
    for spoken in ("m l x", "M-L-X", "M.L.X", "mlx"):
        assert v.apply(spoken) == "MLX"
    v = Vocabulary(ACME_TERMS)
    for spoken in ("a c m l", "A-C-M-L", "A.C.M.L", "acml"):
        assert v.apply(spoken) == "ACML"


def test_multiword_variant_needs_a_separator():
    v = Vocabulary([Term("Wispr Flow", ["whisper flow"])])
    assert v.apply("whisperflow") == "whisperflow"          # NOT matched — no separator
    assert v.apply("whisper-flow") == "Wispr Flow"
    assert v.apply("whisper flow") == "Wispr Flow"


# -- URLs -------------------------------------------------------------------------------

def test_urls_are_not_swallowed():
    v = Vocabulary([Term("AcmeLabs", ["acme labs"]), Term("acmelabs.io", ["acme labs dot io", "acme labs.io"])])
    assert v.apply("el logo de acmelabs.io") == "el logo de acmelabs.io"
    assert v.apply("acme labs dot io") == "acmelabs.io"
    assert v.apply("acme labs.io") == "acmelabs.io"


# -- longest variant wins -----------------------------------------------------------------

def test_longest_variant_wins():
    v = Vocabulary(ACME_TERMS + [Term("Ack-Me", ["ackme"])])
    assert v.apply("the ackme labs customer") == "the Acme Labs customer"
    assert v.apply("the ackme icon") == "the Ack-Me icon"


# -- word boundaries ----------------------------------------------------------------------

def test_word_boundaries_do_not_bleed_into_longer_words():
    v = DEFAULT_VOCAB
    assert v.apply("clodhopper") == "clodhopper"    # "clod" -> "Claude" must not fire here
    assert v.apply("superbases") == "superbases"


# -- accent tolerance on the input ---------------------------------------------------------

def test_accent_tolerance_on_input():
    v = Vocabulary([Term("Café Tacvba", ["cafe tacuba"])])
    assert v.apply("café tacuba") == "Café Tacvba"


# -- stylized casing self-fix ---------------------------------------------------------------

def test_stylized_casing_self_fix():
    v = DEFAULT_VOCAB
    assert v.apply("wispr flow") == "Wispr Flow"
    assert v.apply("manychat") == "ManyChat"


def test_plain_names_are_not_recased():
    v = Vocabulary([Term("Render", [])])
    assert v.apply("render the video") == "render the video"


# -- editing ------------------------------------------------------------------------------

def test_add_merges_case_insensitively_and_dedupes_spoken():
    v = Vocabulary([Term("Foo", ["fu"], "note1")])
    v.add("foo", ["fu2"])
    assert len(v) == 1
    d = v.as_dicts()[0]
    assert d["spoken"] == ["fu", "fu2"]      # merged, no duplicate
    assert d["note"] == "note1"              # no note given -> preserved

    v.add("foo", ["fu2", "fu3"], note="note2")   # re-adding "fu2" doesn't duplicate it
    d = v.as_dicts()[0]
    assert d["spoken"] == ["fu", "fu2", "fu3"]
    assert d["note"] == "note2"              # note given -> updated


def test_add_creates_a_new_term_when_no_match():
    v = Vocabulary([Term("Foo", ["fu"])])
    t = v.add("Brand New", ["bn"], "bn note")
    assert len(v) == 2
    assert t.term == "Brand New" and t.spoken == ["bn"] and t.note == "bn note"


def test_add_rejects_empty_term():
    v = Vocabulary([])
    with pytest.raises(ValueError):
        v.add("   ")


def test_remove_returns_true_when_found_false_otherwise():
    v = Vocabulary([Term("Foo"), Term("Bar")])
    assert v.remove("foo") is True
    assert len(v) == 1
    assert v.remove("doesnotexist") is False
    assert len(v) == 1


def test_replace_all_drops_entries_with_an_empty_term():
    v = Vocabulary([Term("Old")])
    v.replace_all([{"term": "X", "spoken": ["x-spoken"]}, {"term": "", "spoken": ["y"]},
                   {"term": "   ", "spoken": []}])
    assert [t.term for t in v.terms] == ["X"]


# -- persistence ----------------------------------------------------------------------------

def test_load_save_round_trip(tmp_path):
    path = tmp_path / "vocabulary.json"
    v = Vocabulary([Term("A", ["a-spoken"], "notea")])
    v.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.as_dicts() == [{"term": "A", "spoken": ["a-spoken"], "note": "notea", "fuzzy": False, "learned": []}]


def test_load_missing_path_writes_defaults(tmp_path):
    path = tmp_path / "does" / "not" / "exist.json"
    v = Vocabulary.load(path)
    assert path.exists()
    assert len(v) == len(DEFAULT_TERMS)


def test_load_corrupt_file_logs_and_returns_defaults(tmp_path, caplog):
    path = tmp_path / "vocabulary.json"
    path.write_text("{not json")
    v = Vocabulary.load(path)   # must not raise
    assert len(v) == len(DEFAULT_TERMS)


def test_save_leaves_no_tmp_file_behind(tmp_path):
    path = tmp_path / "vocabulary.json"
    Vocabulary([Term("A")]).save(path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


# -- the three consumers ---------------------------------------------------------------------

def test_whisper_terms_unique_ordered_and_capped():
    terms = [Term(f"T{i}") for i in range(MAX_PROMPT_TERMS + 10)]
    terms.append(Term("T0"))          # duplicate of the first term
    v = Vocabulary(terms)
    wt = v.whisper_terms()
    assert len(wt) == MAX_PROMPT_TERMS
    assert len(wt) == len(set(wt))    # unique
    assert wt[0] == "T0" and wt[-1] == f"T{MAX_PROMPT_TERMS - 1}"   # ordered


def test_glossary_mentions_term_note_and_capped_variants():
    v = Vocabulary([Term("Acme Labs", ["acne labs", "ack me labs", "extra variant"],
                         "a fictional company")])
    g = v.glossary()
    assert "Acme Labs" in g
    assert "a fictional company" in g
    assert "acne labs" in g and "ack me labs" in g
    assert "extra variant" not in g   # capped at 2 spoken variants by default


def test_empty_vocabulary_glossary_and_apply_are_noops():
    v = Vocabulary([])
    assert v.glossary() == ""
    assert v.apply("hello world") == "hello world"


# -- a v1 file on disk is migrated once ---------------------------------------------------

def test_v1_file_is_migrated_and_rewritten(tmp_path):
    import json
    from dictdact.vocab import RETIRED_VARIANTS, VOCAB_VERSION
    p = tmp_path / "vocabulary.json"
    p.write_text(json.dumps({"terms": [
        {"term": "Codex", "spoken": ["codecs"], "note": ""},
        {"term": "Hyrox", "spoken": ["hirox", "hierox", "high rocks", "hyrocks"], "note": ""},
    ]}), encoding="utf-8")
    v = Vocabulary.load(p)
    spoken = {t.term: t.spoken for t in v.terms}
    assert spoken["Codex"] == []                                          # the retired seed variant is gone
    assert spoken["Hyrox"] == ["hirox", "hierox", "high rocks", "hyrocks"]  # other seeds are untouched
    assert v.apply("the new codecs; Hirox") == "the new codecs; Hyrox"    # "codecs" is an ordinary word again
    # a seeded term the user customized (any change to its list) is left exactly as it is
    p2 = tmp_path / "custom.json"
    p2.write_text(json.dumps({"terms": [{"term": "Codex", "spoken": ["codecs", "private alias"], "note": ""}]}),
                  encoding="utf-8")
    assert Vocabulary.load(p2).terms[0].spoken == ["codecs", "private alias"]
    # the same spelling on a term the user authored is theirs and stays
    p3 = tmp_path / "mine.json"
    p3.write_text(json.dumps({"terms": [{"term": "Codex", "spoken": ["codecs"], "note": ""},
                                        {"term": "Mine", "spoken": ["codecs", "my own variant"], "note": "user term"}]}),
                  encoding="utf-8")
    v3 = Vocabulary.load(p3)
    assert {t.term: t.spoken for t in v3.terms} == {"Codex": [], "Mine": ["codecs", "my own variant"]}
    assert v3.apply("we replaced the codecs") == "we replaced the Mine"   # the user's mapping survives
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == VOCAB_VERSION
    seeded = {t["term"].lower(): t["spoken"] for t in data["terms"]}
    assert all(s.lower() not in RETIRED_VARIANTS for s in seeded["codex"])
    # loading the rewritten file again changes nothing
    assert Vocabulary.load(p).as_dicts() == v.as_dicts()


def test_revision_is_a_content_hash_stable_across_reloads(tmp_path):
    v = Vocabulary([Term("A", ["a1"])])
    r0 = v.revision
    v.add("B", ["b1"])
    assert v.revision != r0
    r1 = v.revision
    v.remove("B")
    assert v.revision == r0                      # same content again → same revision
    v.replace_all([{"term": "C"}])
    assert v.revision not in (r0, r1)
    # a reload from disk (the CLI path) keeps the revision of the content, not of the object
    p = tmp_path / "vocabulary.json"
    v.save(p)
    assert Vocabulary.load(p).revision == v.revision
    v.add("D")
    v.save(p)
    assert Vocabulary.load(p).revision != Vocabulary([Term("C")]).revision


def test_concurrent_saves_never_collide_or_leave_temp_files(tmp_path):
    import threading
    p = tmp_path / "vocabulary.json"
    errors = []

    def writer(i):
        try:
            Vocabulary([Term(f"T{i}")]).save(p)
        except Exception as e:      # pragma: no cover - the assertion below reports it
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert Vocabulary.load(p).terms[0].term.startswith("T")
    assert [x.name for x in tmp_path.iterdir() if x.name.endswith(".tmp")] == []


def test_vocab_lock_is_reentrant_in_process(tmp_path):
    from dictdact.vocab import vocab_lock
    p = tmp_path / "vocabulary.json"
    with vocab_lock(p):
        with vocab_lock(p):
            Vocabulary([Term("A")]).save(p)      # save() locks again — must not deadlock
        assert Vocabulary.load(p).terms[0].term == "A"


def test_vocab_lock_excludes_other_threads(tmp_path):
    import threading, time
    from dictdact.vocab import vocab_lock
    p = tmp_path / "vocabulary.json"
    inside, overlap = [], []
    gate = threading.Event()

    def worker(name):
        with vocab_lock(p):
            if inside:
                overlap.append(name)          # someone else is already in the critical section
            inside.append(name)
            gate.set()
            time.sleep(0.05)
            inside.remove(name)

    a = threading.Thread(target=worker, args=("a",)); b = threading.Thread(target=worker, args=("b",))
    a.start(); gate.wait(1.0); b.start(); a.join(); b.join()
    assert overlap == []


def test_apply_uses_one_snapshot_while_the_list_changes():
    import threading
    v = Vocabulary([Term("AAA", ["aaa"]), Term("BBB", ["bbb"])])
    text = " ".join(["aaa bbb xxx yyy"] * 200)
    stop = threading.Event()
    errors = []

    def mutate():
        i = 0
        while not stop.is_set():
            v.replace_all([{"term": "XXX", "spoken": ["xxx"]}, {"term": "YYY", "spoken": ["yyy"]}] if i % 2 else
                          [{"term": "AAA", "spoken": ["aaa"]}, {"term": "BBB", "spoken": ["bbb"]}])
            i += 1

    t = threading.Thread(target=mutate); t.start()
    try:
        for _ in range(300):
            try:
                out = v.apply(text)
            except Exception as e:      # a half-swapped regex/group table used to raise KeyError
                errors.append(e); break
            has_ab = "AAA" in out or "BBB" in out
            has_xy = "XXX" in out or "YYY" in out
            assert not (has_ab and has_xy), "one take was replaced with two dictionary revisions"
    finally:
        stop.set(); t.join()
    assert errors == []


def test_v2_file_gets_the_seed_fuzzy_flags(tmp_path):
    import json
    from dictdact.vocab import VOCAB_VERSION
    p = tmp_path / "vocabulary.json"
    p.write_text(json.dumps({"version": 2, "terms": [
        {"term": "Chatterbox", "spoken": [], "note": ""},                    # a seed that ships with fuzzy off
        {"term": "Hyrox", "spoken": ["hirox"], "note": ""},                  # a seed that ships with fuzzy on
        {"term": "MyTool", "spoken": [], "note": ""},                        # the user's own term: stays off
    ]}), encoding="utf-8")
    v = Vocabulary.load(p)
    flags = {t.term: t.fuzzy for t in v.terms}
    assert flags == {"Chatterbox": False, "Hyrox": True, "MyTool": False}  # a user's own term: opt-in
    assert json.loads(p.read_text(encoding="utf-8"))["version"] == VOCAB_VERSION
