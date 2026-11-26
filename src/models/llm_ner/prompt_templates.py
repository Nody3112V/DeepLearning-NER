"""Prompt templates for LLM-based Vietnamese NER on PhoNER_COVID19.

Type definitions follow Table 1 and the appendix notes of Truong et al. (2021),
"COVID-19 Named Entity Recognition for Vietnamese" (arXiv 2104.03879).

The sentence is shown as `sentence_text(tokens)` (tokens joined by single
spaces, underscores of word-level tokens turned into spaces). postprocess.py
searches the LLM's `text` in exactly that string, so keep the two in sync.

Bump PROMPT_VERSION whenever the wording changes: it is part of the cache key.
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

try:  # package layout (python -m src.models.llm_ner.run_zeroshot)
    from .postprocess import bio_to_entities, sentence_text
except ImportError:  # flat layout
    from postprocess import bio_to_entities, sentence_text

PROMPT_VERSION = "zs-v1"

# TODO(verify on the train split before trusting the examples below): span
# conventions that the paper does not state, e.g. whether AGE includes "tuổi"
# ("28" vs "28 tuổi") and whether DATE includes "ngày".
PHONER_ENTITY_TYPES: dict[str, str] = {
    "PATIENT_ID": (
        "Identifier of a COVID-19 patient in Vietnam. Tag only the patient number/code "
        "(e.g. \"669\" in \"bệnh nhân 669\"), not the words \"bệnh nhân\"."
    ),
    "PERSON_NAME": "Name of a patient or of a person who came into contact with a patient.",
    "AGE": (
        "Age of a patient or contact person. Tag it only when the sentence links it to a "
        "specific named person or patient ID (e.g. \"28 tuổi\")."
    ),
    "GENDER": (
        "Gender of a patient or contact person. Tag it only when linked to a specific named "
        "person or patient ID (e.g. \"nam\", \"nữ\")."
    ),
    "OCCUPATION": (
        "Job of a specific named person or patient ID in the same sentence (e.g. \"bác sĩ\" in "
        "\"bệnh nhân 669 là bác sĩ\"). Do not tag a generic group (e.g. \"điều dưỡng\") that is "
        "not linked to a specific person."
    ),
    "LOCATION": (
        "Places where a patient was present: addresses, wards, districts, provinces, and "
        "hospitals or other organizations named as the place a patient was at. Do not tag "
        "generic place mentions such as \"bệnh viện tỉnh\", \"trạm y tế xã\", \"chung cư\"."
    ),
    "ORGANIZATION": (
        "Organizations related to a patient (company, government body, hospital, ...) when "
        "they act as an agent, e.g. a hospital that reports a test result, rather than as the "
        "place a patient was at."
    ),
    "SYMPTOM_AND_DISEASE": (
        "Symptoms a patient experiences, diseases a patient had before COVID-19, and "
        "complications mentioned in death reports. Make the span as detailed as possible, "
        "including severity words (e.g. \"viêm phổi nặng\", \"suy thận mạn giai đoạn cuối\")."
    ),
    "TRANSPORTATION": (
        "Specific vehicle identifiers only: flight numbers and bus/car plates. Do not tag "
        "generic words such as \"máy bay\" or \"xe khách\"."
    ),
    "DATE": "Any date that appears in the sentence (e.g. \"12/3\").",
}

TYPE_SETS: dict[str, Mapping[str, str]] = {"phoner": PHONER_ENTITY_TYPES}

SYSTEM_INSTRUCTION = (
    "You are an expert annotator of Vietnamese named entities in COVID-19 news sentences. "
    "You follow the annotation guidelines exactly and answer with JSON only."
)

_RULES = """\
Rules:
1. Use only the sentence below. Copy each entity's "text" exactly as it appears in the \
sentence (same characters, diacritics and spacing). Never translate, correct or invent text.
2. Entities do not overlap and are not nested. When a longer span is an entity, return the \
longer span only.
3. List each distinct entity once, even if it occurs several times in the sentence.
4. Use only the types listed above. If the sentence has no entity, return an empty list."""

_FORMAT = """\
Return one JSON object and nothing else (no explanation, no markdown):
{"entities": [{"text": "<exact text from the sentence>", "type": "<ONE OF THE TYPES>"}]}"""


def _render_types(types: Mapping[str, str], include_guidelines: bool) -> str:
    if include_guidelines:
        return "\n".join(f"- {name}: {desc}" for name, desc in types.items())
    return "\n".join(f"- {name}" for name in types)


def _render_example(ex: Mapping[str, Sequence[str]]) -> str:
    entities = bio_to_entities(ex["tokens"], ex["tags"])
    out = json.dumps({"entities": entities}, ensure_ascii=False)
    return f"Sentence: {sentence_text(ex['tokens'])}\nOutput: {out}"


def build_prompt(
    tokens: Sequence[str],
    *,
    examples: Sequence[Mapping[str, Sequence[str]]] | None = None,
    entity_types: Mapping[str, str] | None = None,
    include_guidelines: bool = True,
) -> str:
    """User prompt for one sentence.

    examples: optional few-shot demonstrations, each {"tokens": [...], "tags": [...]};
        rendered with the same JSON format the model must produce.
    include_guidelines: False lists type names only (ablation).
    """
    types = entity_types or PHONER_ENTITY_TYPES
    parts = [
        "Task: find all named entities in the Vietnamese sentence and label each with one "
        "of the types below.",
        "Entity types:\n" + _render_types(types, include_guidelines),
        _RULES,
        _FORMAT,
    ]
    if examples:
        parts.append("Examples:\n" + "\n\n".join(_render_example(e) for e in examples))
    parts.append(f"Sentence: {sentence_text(tokens)}\nOutput:")
    return "\n\n".join(parts)


def build_messages(tokens: Sequence[str], **kwargs) -> tuple[str, str]:
    """(system_instruction, user_prompt)."""
    return SYSTEM_INSTRUCTION, build_prompt(tokens, **kwargs)
