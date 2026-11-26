"""LLM entity list -> BIO tags aligned with the input tokens.

Pipeline (one sentence):
    raw LLM text --parse_llm_response--> list of {"text", "type"} items
                 --entities_to_bio-----> BIO tags, len(tags) == len(tokens)

Design rules
------------
* The output ALWAYS has exactly len(tokens) tags, whatever the LLM returned.
* Matching works on a normalised, space-joined version of the sentence
  (`sentence_text`). The prompt shows the LLM that same string, so the LLM's
  `text` is normally an exact substring of it. Underscores (word-level tokens
  such as "Hà_Nội") are turned into spaces, so syllable- and word-level data
  are handled by the same code. A match must start and end on token
  boundaries, otherwise it is rejected as "misaligned".
* Every aligned occurrence of an entity text is labelled (`repeat="all"`,
  PhoNER annotates every mention). Use `repeat="first"` for the alternative.
* Overlaps are resolved greedily: longest span first, then LLM order. PhoNER
  has no nested entities, so the shorter span is dropped.
* Every failure mode is counted in `PostprocessStats` (Week 4 needs the
  malformed-output rate).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable, Sequence

__all__ = [
    "PostprocessResult",
    "PostprocessStats",
    "ParseResult",
    "bio_to_entities",
    "entities_to_bio",
    "is_valid_bio",
    "parse_llm_response",
    "postprocess",
    "sentence_text",
]

_WS = re.compile(r"\s+")
_PLACEHOLDER = "\u2400"  # stands in for a token whose normalised text is empty
_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*|\s*```\s*$")
_LIST_KEYS = ("entities", "named_entities", "results", "result")
_TEXT_KEYS = ("text", "entity", "mention", "span")
_TYPE_KEYS = ("type", "label", "category", "entity_type")


# --------------------------------------------------------------------------
# Text normalisation (shared with prompt_templates via sentence_text)
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("_", " ")
    return _WS.sub(" ", s).strip()


def _norm_tokens(tokens: Sequence[str]) -> list[str]:
    out = []
    for t in tokens:
        n = _norm(str(t))
        out.append(n if n else _PLACEHOLDER)
    return out


def sentence_text(tokens: Sequence[str]) -> str:
    """The exact string shown to the LLM and searched by `entities_to_bio`."""
    return " ".join(_norm_tokens(tokens))


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
@dataclass
class PostprocessStats:
    """Counters; sentence-level unless noted. Sum them with `merge`."""

    n_sentences: int = 0
    n_malformed: int = 0          # response could not be turned into an entity list
    n_repaired: int = 0           # parsed only after fence-stripping / extraction
    n_items: int = 0              # entity items in the parsed list (entity-level)
    n_items_invalid: int = 0      # not a dict / missing text or type
    n_invalid_type: int = 0       # type not in the allowed set
    n_duplicates: int = 0         # same (text, type) listed again
    n_checked: int = 0            # items that reached the matching stage
    n_hallucinated: int = 0       # text not found anywhere in the sentence
    n_misaligned: int = 0         # found, but not on token boundaries
    n_case_insensitive: int = 0   # matched only after ignoring case
    n_matched: int = 0            # items with >= 1 aligned occurrence
    n_spans_labeled: int = 0      # spans written to the tags (occurrence-level)
    n_spans_dropped_overlap: int = 0   # occurrence-level
    n_type_conflicts: int = 0     # same span, different type; first listed wins
    malformed_reasons: dict = field(default_factory=dict)

    def merge(self, other: "PostprocessStats") -> "PostprocessStats":
        for f in fields(self):
            if f.name == "malformed_reasons":
                for k, v in other.malformed_reasons.items():
                    self.malformed_reasons[k] = self.malformed_reasons.get(k, 0) + v
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def malformed_rate(self) -> float:
        return self._rate(self.n_malformed, self.n_sentences)

    @property
    def hallucination_rate(self) -> float:
        return self._rate(self.n_hallucinated, self.n_checked)

    @property
    def unmatched_rate(self) -> float:
        return self._rate(self.n_hallucinated + self.n_misaligned, self.n_checked)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["malformed_rate"] = self.malformed_rate
        d["hallucination_rate"] = self.hallucination_rate
        d["unmatched_rate"] = self.unmatched_rate
        return d


@dataclass
class PostprocessResult:
    tags: list
    status: str                       # "ok" | "malformed"
    stats: PostprocessStats
    entities: list = field(default_factory=list)  # matched spans (token idx, inclusive)
    issues: list = field(default_factory=list)    # [{"kind": ..., "text": ...}]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
@dataclass
class ParseResult:
    items: list
    ok: bool
    repaired: bool = False
    reason: str | None = None  # "empty_response" | "no_json" | "bad_structure"


def _loads_lenient(raw: str) -> tuple[Any, bool]:
    """Return (object or None, repaired?)."""
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass
    stripped = _FENCE.sub("", raw).strip()
    try:
        return json.loads(stripped), True
    except json.JSONDecodeError:
        pass
    # Last resort: first decodable object/array inside surrounding prose.
    # NOTE: on a truncated array this salvages only the first complete object.
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[m.start():])
            return obj, True
        except json.JSONDecodeError:
            continue
    return None, False


def _first_key(d: dict, keys: Sequence[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _item_fields(item: Any) -> tuple[str | None, str | None]:
    if not isinstance(item, dict):
        return None, None
    text = _first_key(item, _TEXT_KEYS)
    typ = _first_key(item, _TYPE_KEYS)
    return (text if isinstance(text, str) else None,
            typ if isinstance(typ, str) else None)


def _extract_items(obj: Any) -> list | None:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in _LIST_KEYS:
            if k in obj:
                v = obj[k]
                if v is None:
                    return []
                return v if isinstance(v, list) else None
        if _item_fields(obj) != (None, None):  # a single bare entity
            return [obj]
    return None


def parse_llm_response(raw: str | None) -> ParseResult:
    """Raw model output -> list of (unvalidated) entity items."""
    if raw is None or not str(raw).strip():
        return ParseResult([], False, reason="empty_response")
    obj, repaired = _loads_lenient(str(raw).strip())
    if obj is None:
        return ParseResult([], False, reason="no_json")
    items = _extract_items(obj)
    if items is None:
        return ParseResult([], False, repaired, reason="bad_structure")
    return ParseResult(items, True, repaired)


# --------------------------------------------------------------------------
# Entity items -> BIO
# --------------------------------------------------------------------------
def _type_key(raw: str) -> str:
    t = raw.strip().upper().replace("&", "_AND_")
    return re.sub(r"[^A-Z0-9]+", "_", t).strip("_")


def _canonical_types(allowed: Iterable[str]) -> dict[str, str]:
    return {_type_key(a): a for a in allowed}


def _lookup_type(raw: str, canon: dict[str, str]) -> str | None:
    key = _type_key(raw)
    if key in canon:
        return canon[key]
    if key[:2] in ("B_", "I_") and key[2:] in canon:  # model echoed a BIO prefix
        return canon[key[2:]]
    return None


def _find_aligned(hay: str, needle: str, start_of: dict, end_of: dict) -> list:
    """All occurrences of needle that start and end on token boundaries."""
    out = []
    k = hay.find(needle)
    while k != -1:
        e = k + len(needle)
        if k in start_of and e in end_of:
            out.append((start_of[k], end_of[e]))
        k = hay.find(needle, k + 1)
    return out


def entities_to_bio(
    tokens: Sequence[str],
    items: Iterable[Any],
    allowed_types: Iterable[str],
    *,
    repeat: str = "all",
    case_fallback: bool = True,
) -> tuple[list, PostprocessStats, list, list]:
    """Return (tags, stats, matched_entities, issues). `stats.n_sentences` stays 0."""
    if repeat not in ("all", "first"):
        raise ValueError("repeat must be 'all' or 'first'")
    stats = PostprocessStats()
    issues: list = []
    n = len(tokens)
    tags = ["O"] * n
    if n == 0:
        return tags, stats, [], issues

    canon = _canonical_types(allowed_types)
    norm_toks = _norm_tokens(tokens)
    text = " ".join(norm_toks)
    start_of: dict[int, int] = {}
    end_of: dict[int, int] = {}
    pos = 0
    for i, nt in enumerate(norm_toks):
        if i:
            pos += 1  # the single joining space
        start_of[pos] = i
        pos += len(nt)
        end_of[pos] = i
    lowered = text.lower()
    text_lc = lowered if (case_fallback and len(lowered) == len(text)) else None

    # 1) validate and de-duplicate items
    entities: list[tuple[int, str, str]] = []  # (order, needle, type)
    seen: set = set()
    for item in items:
        stats.n_items += 1
        txt, typ_raw = _item_fields(item)
        needle = _norm(txt) if txt is not None else ""
        if not needle or typ_raw is None:
            stats.n_items_invalid += 1
            issues.append({"kind": "invalid_item", "text": str(item)[:80]})
            continue
        typ = _lookup_type(typ_raw, canon)
        if typ is None:
            stats.n_invalid_type += 1
            issues.append({"kind": "invalid_type", "text": f"{needle} [{typ_raw}]"})
            continue
        if (needle, typ) in seen:
            stats.n_duplicates += 1
            continue
        seen.add((needle, typ))
        entities.append((len(entities), needle, typ))
    stats.n_checked = len(entities)

    # 2) locate every entity in the sentence
    spans: list[tuple[int, int, int, str, str]] = []  # (order, i, j, type, needle)
    for order, needle, typ in entities:
        occ = _find_aligned(text, needle, start_of, end_of)
        case_only = False
        if not occ and text_lc is not None:
            occ = _find_aligned(text_lc, needle.lower(), start_of, end_of)
            case_only = bool(occ)
        if not occ:
            present = needle in text or (text_lc is not None and needle.lower() in text_lc)
            if present:
                stats.n_misaligned += 1
                issues.append({"kind": "misaligned", "text": needle})
            else:
                stats.n_hallucinated += 1
                issues.append({"kind": "hallucinated", "text": needle})
            continue
        if case_only:
            stats.n_case_insensitive += 1
        stats.n_matched += 1
        for (i, j) in (occ[:1] if repeat == "first" else occ):
            spans.append((order, i, j, typ, needle))

    # 3) resolve overlaps: longest first, then LLM order, then position
    spans.sort(key=lambda s: (-(s[2] - s[1] + 1), s[0], s[1]))
    owner: list = [None] * n
    accepted: list[tuple[int, int, str, str]] = []
    for _order, i, j, typ, needle in spans:
        clash = {owner[t] for t in range(i, j + 1) if owner[t] is not None}
        if clash:
            same = [accepted[c] for c in clash if accepted[c][:2] == (i, j)]
            if same and same[0][2] == typ:
                stats.n_duplicates += 1
            elif same:
                stats.n_type_conflicts += 1
                issues.append({"kind": "type_conflict", "text": needle})
            else:
                stats.n_spans_dropped_overlap += 1
                issues.append({"kind": "overlap_dropped", "text": needle})
            continue
        idx = len(accepted)
        accepted.append((i, j, typ, needle))
        for t in range(i, j + 1):
            owner[t] = idx

    for i, j, typ, _needle in accepted:
        tags[i] = f"B-{typ}"
        for t in range(i + 1, j + 1):
            tags[t] = f"I-{typ}"
    stats.n_spans_labeled = len(accepted)
    matched = [
        {"text": needle, "type": typ, "start": i, "end": j}
        for i, j, typ, needle in sorted(accepted)
    ]
    return tags, stats, matched, issues


def postprocess(
    tokens: Sequence[str],
    raw: str | None,
    allowed_types: Iterable[str],
    *,
    repeat: str = "all",
    case_fallback: bool = True,
    failure_reason: str | None = None,
) -> PostprocessResult:
    """Raw LLM response for one sentence -> BIO tags (always len(tokens)).

    failure_reason: set by the caller when the API says the response is unusable
    (e.g. "finish_max_tokens": cut off, so the JSON must not be salvaged). The
    sentence is then counted as malformed under that reason.
    """
    allowed_types = list(allowed_types)
    if failure_reason:
        parsed = ParseResult([], False, reason=failure_reason)
    else:
        parsed = parse_llm_response(raw)
    if not parsed.ok:
        stats = PostprocessStats(n_sentences=1, n_malformed=1,
                                 malformed_reasons={parsed.reason: 1})
        return PostprocessResult(
            tags=["O"] * len(tokens), status="malformed", stats=stats,
            issues=[{"kind": "malformed", "text": parsed.reason}],
        )
    tags, ent_stats, matched, issues = entities_to_bio(
        tokens, parsed.items, allowed_types, repeat=repeat, case_fallback=case_fallback
    )
    stats = PostprocessStats(n_sentences=1, n_repaired=int(parsed.repaired))
    stats.merge(ent_stats)
    return PostprocessResult(tags=tags, status="ok", stats=stats,
                             entities=matched, issues=issues)


# --------------------------------------------------------------------------
# BIO -> entity items (few-shot examples, oracle tests)
# --------------------------------------------------------------------------
def bio_to_entities(tokens: Sequence[str], tags: Sequence[str], *, dedupe: bool = True) -> list:
    """Gold BIO tags -> [{"text", "type"}] in order of appearance."""
    norm_toks = _norm_tokens(tokens)
    out: list = []
    start = None
    cur = None

    def close(end: int):
        if start is not None:
            out.append({"text": " ".join(norm_toks[start:end + 1]), "type": cur})

    for i, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close(i - 1)
            start, cur = None, None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or start is None or typ != cur:
            close(i - 1)
            start, cur = i, typ
    close(len(tags) - 1)
    if dedupe:
        seen, uniq = set(), []
        for e in out:
            k = (e["text"], e["type"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        out = uniq
    return out


def is_valid_bio(tags: Sequence[str]) -> bool:
    """True if no I-X follows anything other than B-X / I-X."""
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            if prev == "O" or prev[2:] != t[2:]:
                return False
        prev = t
    return True
