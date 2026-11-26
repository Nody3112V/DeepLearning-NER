"""Run from the repo root:  python -m pytest tests/test_postprocess.py"""
import json
import random

import pytest

from src.models.llm_ner.postprocess import (
    PostprocessStats,
    bio_to_entities,
    is_valid_bio,
    parse_llm_response,
    postprocess,
    sentence_text,
)
from src.models.llm_ner.prompt_templates import PHONER_ENTITY_TYPES, build_prompt

TYPES = list(PHONER_ENTITY_TYPES)


def J(*ents):
    return json.dumps({"entities": [{"text": t, "type": ty} for t, ty in ents]}, ensure_ascii=False)


SYL = ["Bệnh", "nhân", "669", "ở", "Hà", "Nội", "và", "Hà", "Nội"]
WORD = ["Bệnh_nhân", "669", "ở", "Hà_Nội", "và", "Hà_Nội"]


def test_syllable_level_alignment():
    r = postprocess(SYL, J(("669", "PATIENT_ID"), ("Hà Nội", "LOCATION")), TYPES)
    assert r.status == "ok"
    assert r.tags == ["O", "O", "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION",
                      "O", "B-LOCATION", "I-LOCATION"]


def test_word_level_underscores_either_form():
    for text in ("Hà Nội", "Hà_Nội"):
        r = postprocess(WORD, J((text, "LOCATION")), TYPES)
        assert r.tags == ["O", "O", "O", "B-LOCATION", "O", "B-LOCATION"]


def test_repeat_policy():
    ent = J(("Hà Nội", "LOCATION"))
    assert postprocess(WORD, ent, TYPES, repeat="all").tags.count("B-LOCATION") == 2
    assert postprocess(WORD, ent, TYPES, repeat="first").tags.count("B-LOCATION") == 1


def test_hallucinated_and_misaligned_are_counted():
    r = postprocess(WORD, J(("Đà Nẵng", "LOCATION")), TYPES)
    assert r.stats.n_hallucinated == 1 and set(r.tags) == {"O"}
    r = postprocess(WORD, J(("Nội", "LOCATION")), TYPES)  # inside a word-level token
    assert r.stats.n_misaligned == 1 and r.stats.n_hallucinated == 0
    assert r.stats.unmatched_rate == 1.0


def test_overlap_longest_wins():
    toks = ["Bệnh", "viện", "Bạch", "Mai", "đóng", "cửa"]
    r = postprocess(toks, J(("Bạch Mai", "LOCATION"), ("Bệnh viện Bạch Mai", "ORGANIZATION")), TYPES)
    assert r.tags[:4] == ["B-ORGANIZATION", "I-ORGANIZATION", "I-ORGANIZATION", "I-ORGANIZATION"]
    assert r.stats.n_spans_dropped_overlap == 1


def test_type_conflict_first_listed_wins():
    r = postprocess(["Hà", "Nội"], J(("Hà Nội", "ORGANIZATION"), ("Hà Nội", "LOCATION")), TYPES)
    assert r.tags == ["B-ORGANIZATION", "I-ORGANIZATION"]
    assert r.stats.n_type_conflicts == 1


def test_type_normalisation_and_invalid_type():
    toks = ["viêm", "phổi", "nặng"]
    for ty in ("SYMPTOM&DISEASE", "symptom & disease", "B-SYMPTOM_AND_DISEASE"):
        assert postprocess(toks, J(("viêm phổi nặng", ty)), TYPES).tags[0] == "B-SYMPTOM_AND_DISEASE"
    r = postprocess(toks, J(("viêm phổi nặng", "DISEASE")), TYPES)
    assert r.stats.n_invalid_type == 1 and set(r.tags) == {"O"}


def test_case_insensitive_fallback():
    r = postprocess(WORD, J(("hà nội", "LOCATION")), TYPES)
    assert r.tags.count("B-LOCATION") == 2 and r.stats.n_case_insensitive == 1
    r = postprocess(WORD, J(("hà nội", "LOCATION")), TYPES, case_fallback=False)
    assert r.stats.n_hallucinated == 1


def test_duplicates():
    r = postprocess(WORD, J(("Hà Nội", "LOCATION"), ("Hà Nội", "LOCATION")), TYPES)
    assert r.stats.n_duplicates == 1 and r.tags.count("B-LOCATION") == 2


@pytest.mark.parametrize("raw,reason", [
    (None, "empty_response"), ("", "empty_response"), ("  \n", "empty_response"),
    ("Sorry, I can't do that.", "no_json"), ('{"foo": 1}', "bad_structure"),
    ('{"entities": "none"}', "bad_structure"), ("42", "bad_structure"),
])
def test_malformed(raw, reason):
    r = postprocess(WORD, raw, TYPES)
    assert r.status == "malformed" and r.tags == ["O"] * len(WORD)
    assert r.stats.n_malformed == 1 and r.stats.malformed_reasons == {reason: 1}


def test_repairs_and_shapes():
    good = J(("669", "PATIENT_ID"))
    for raw in ("```json\n" + good + "\n```", "Here you go: " + good + " Hope it helps"):
        r = postprocess(WORD, raw, TYPES)
        assert r.status == "ok" and r.stats.n_repaired == 1 and r.tags[1] == "B-PATIENT_ID"
    assert postprocess(WORD, good, TYPES).stats.n_repaired == 0
    bare = json.dumps([{"text": "669", "type": "PATIENT_ID"}])
    assert postprocess(WORD, bare, TYPES).tags[1] == "B-PATIENT_ID"
    alias = json.dumps({"entities": [{"entity": "669", "label": "PATIENT_ID"}]})
    assert postprocess(WORD, alias, TYPES).tags[1] == "B-PATIENT_ID"
    assert postprocess(WORD, '{"entities": []}', TYPES).status == "ok"
    assert postprocess(WORD, '{"entities": null}', TYPES).status == "ok"


def test_invalid_items_do_not_crash():
    raw = json.dumps({"entities": [1, "x", {"text": "669"}, {"type": "AGE"}, {"text": 5, "type": "AGE"},
                                   {"text": "669", "type": "PATIENT_ID"}]})
    r = postprocess(WORD, raw, TYPES)
    assert r.stats.n_items == 6 and r.stats.n_items_invalid == 5
    assert r.tags[1] == "B-PATIENT_ID"


def test_empty_tokens():
    r = postprocess([], J(("x", "AGE")), TYPES)
    assert r.tags == [] and r.status == "ok"


def test_stats_merge_and_rates():
    a = postprocess(WORD, "nope", TYPES).stats
    b = postprocess(WORD, J(("Hà Nội", "LOCATION"), ("Đà Nẵng", "LOCATION")), TYPES).stats
    total = PostprocessStats().merge(a).merge(b)
    d = total.to_dict()
    assert d["n_sentences"] == 2 and d["malformed_rate"] == 0.5
    assert d["n_checked"] == 2 and d["hallucination_rate"] == 0.5
    assert d["malformed_reasons"] == {"no_json": 1}


def test_bio_roundtrip_oracle():
    gold = ["O", "O", "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION", "O", "O", "O"]
    ents = bio_to_entities(SYL, gold)
    assert ents == [{"text": "669", "type": "PATIENT_ID"}, {"text": "Hà Nội", "type": "LOCATION"}]
    # the second "Hà Nội" mention is labelled too (repeat="all"), so tags differ from gold
    r = postprocess(SYL, json.dumps({"entities": ents}, ensure_ascii=False), TYPES)
    assert r.tags[4:6] == gold[4:6] and r.tags[7:9] == ["B-LOCATION", "I-LOCATION"]


def test_random_invariants():
    rng = random.Random(0)
    vocab = ["Hà", "Nội", "bệnh_nhân", "669", "ở", "và", "(", ")", "Bạch", "Mai", "_", "  "]
    junk = ["", None, "[]", "{", '{"entities": [{"text": "Hà", "type": "LOCATION"}', "```", "[1,2]"]
    for _ in range(300):
        toks = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        ents = [{"text": " ".join(rng.sample(vocab, rng.randint(1, 3))), "type": rng.choice(TYPES + ["X"])}
                for _ in range(rng.randint(0, 5))]
        for raw in (json.dumps({"entities": ents}, ensure_ascii=False), rng.choice(junk)):
            r = postprocess(toks, raw, TYPES)
            assert len(r.tags) == len(toks)
            assert is_valid_bio(r.tags)


def test_prompt_matches_postprocess_text():
    toks = ["Bệnh_nhân", "669", "ở", "Hà_Nội"]
    p = build_prompt(toks)
    assert sentence_text(toks) == "Bệnh nhân 669 ở Hà Nội"
    assert "Sentence: Bệnh nhân 669 ở Hà Nội\nOutput:" in p
    for t in TYPES:
        assert t in p
    shot = build_prompt(toks, examples=[{"tokens": toks, "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"]}])
    assert '"text": "669", "type": "PATIENT_ID"' in shot
