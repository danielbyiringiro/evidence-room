"""
Guards on the two invariants that would fail silently.

Access leakage produces a plausible-looking citation to a judgment made about a
different question, and missing provenance produces a suggestion that cannot be
traced back to an approved rubric. Both look fine in a demo.

Run: pytest -q  (from repo root, with the corpus present)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_room.ingest import (  # noqa: E402
    DATA, RUBRIC_ITEMS, SOURCES, access_filter, build_index, chunk_rubric_guidance,
)

corpus_present = (DATA / "Grading Instructions.csv").exists()
needs_corpus = pytest.mark.skipif(not corpus_present, reason="corpus not present; see data/README.md")


def test_rubric_bank_has_all_seven_items():
    chunks = chunk_rubric_guidance()
    assert len(chunks) == 7, f"expected 7 rubric items, got {len(chunks)}"
    assert len({c.chunk_id for c in chunks}) == 7, "rubric chunk ids not unique"


def test_rubric_chunks_carry_general_rules():
    """Caveats must travel with the deduction they govern."""
    for c in chunk_rubric_guidance():
        assert "General rules that also apply:" in c.text
        assert len(c.text.split()) > 50, f"{c.chunk_id} suspiciously thin"


@needs_corpus
def test_no_cross_question_leakage():
    chunks = build_index(max_per_question=25)
    for _, _, qkey in SOURCES:
        visible = access_filter(chunks, qkey)
        leaked = [c for c in visible if c.question_key not in (None, qkey)]
        assert not leaked, f"{qkey}: {len(leaked)} chunks leaked across questions"


@needs_corpus
def test_rubric_guidance_is_shared_not_scoped():
    """Guidance is the same approved rubric for all four questions."""
    chunks = build_index(max_per_question=10)
    for _, _, qkey in SOURCES:
        shared = [c for c in access_filter(chunks, qkey) if c.doc_type == "rubric_guidance"]
        assert len(shared) == 7, f"{qkey} sees {len(shared)} rubric chunks, expected 7"


@needs_corpus
def test_every_exemplar_is_citable():
    """A suggestion with no traceable source is worse than no suggestion."""
    required = {"submission_id", "question_key", "rubric_item", "score", "label"}
    for c in build_index(max_per_question=25):
        if c.doc_type != "graded_exemplar":
            continue
        assert required <= set(c.provenance), f"{c.chunk_id} missing {required - set(c.provenance)}"
        assert c.provenance["score"] in (0, 1, 2)


@needs_corpus
def test_chunk_ids_unique():
    chunks = build_index(max_per_question=25)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate chunk ids"


@needs_corpus
def test_exemplars_split_per_rubric_item():
    """One chunk per (submission x item) -- not one per submission."""
    chunks = [c for c in build_index(max_per_question=5) if c.doc_type == "graded_exemplar"]
    by_sub: dict[str, set[str]] = {}
    for c in chunks:
        by_sub.setdefault(c.provenance["submission_id"], set()).add(c.provenance["rubric_column"])
    assert by_sub, "no exemplars produced"
    for sub, cols in by_sub.items():
        assert cols <= set(RUBRIC_ITEMS), f"{sub} has unknown rubric columns"
