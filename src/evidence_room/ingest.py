"""
Evidence Room ingestion -- real corpus.

Source: UIUC written-induction dataset (Dataverse), 3,597 expert-graded student
proofs across 4 induction questions, scored against a 7-item rubric.

Two document families get indexed, and they play different roles:

  1. RUBRIC GUIDANCE (from "Grading Instructions.csv")
     This is the deduction bank. Each rubric item carries the grader's written
     reasoning about edge cases -- when partial credit applies, what does not
     earn credit, how to treat implicit work. This is what gets RETRIEVED as
     evidence for a suggested deduction.

  2. GRADED EXEMPLARS (from the four submission CSVs)
     Real student proofs with per-item expert labels. These are retrieved as
     "here is a proof previously judged this way on this rubric item", which is
     the few-shot-by-retrieval pattern Zhao et al. found beat random selection.

Chunking stays TYPE-AWARE (same rationale as the synthetic pass): a rubric item
and a graded exemplar are different units of evidence and must never be merged
into one chunk.

Label encoding in the source CSVs: 0 = not present (n), 1 = partial (p), 2 = correct (c).
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import csv
import json
import os
import re

import pandas as pd

DATA = Path(os.environ.get("EVIDENCE_ROOM_DATA", Path(__file__).resolve().parents[2] / "data"))

# Column name in the CSVs -> (human rubric name, proof section)
RUBRIC_ITEMS: dict[str, tuple[str, str]] = {
    "Identify.Base.Case": ("Identify Base Case", "Base case"),
    "Prove.Base.Case": ("Prove Base Case", "Base case"),
    "Hypothesis.is.stated": ("Hypothesis is stated", "Inductive Hypothesis"),
    "Hypothesis.is.given.some.bound": ("Hypothesis is given some bound", "Inductive Hypothesis"),
    "Goal.is.Clear": ("Goal is clear", "Inductive Step"),
    "Expression.of.Size.k.1.is.decomposed.into.expression.of.size.k": (
        "Expression of Size k+1 is decomposed into expression of size k", "Inductive Step"),
    "Inductive.Hypothesis.is.applied": ("Inductive Hypothesis is applied", "Inductive Step"),
}

LABELS = {0: "not present", 1: "partial", 2: "correct"}

# Each submission CSV -> the question directory holding question.html
SOURCES = [
    ("WrittenInduction_v1-all.csv", "WrittenInduction_v1", "sum-formula"),
    ("WrittenInduction_Sum_v3-sp23.csv", "WrittenInduction_Sum_v3", "factorial-sum"),
    ("WrittenInduction_Recurrence_v3-all.csv", "WrittenInduction_Recurrence_v3", "recurrence"),
    ("WrittenInduction_Divisibilty_v4-sp23.csv", "WrittenInduction_Divisibilty_v4", "divisibility"),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_type: str            # "rubric_guidance" | "graded_exemplar"
    text: str                # embedded text
    question_key: str | None # access / scoping dimension
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def read_question(dirname: str) -> str:
    """Strip HTML/template noise from question.html to get the claim being proved."""
    raw = (DATA / dirname / "question.html").read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\{\{.*?\}\}", " ", raw, flags=re.S)   # mustache template blocks
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_rubric_guidance() -> list[Chunk]:
    """
    One chunk per rubric item, carrying its written grading rationale.

    The generic notes at the bottom of the sheet ("a header alone earns no
    credit", "out-of-order statements still count") apply to EVERY item, so they
    are appended to each chunk rather than indexed separately -- a retrieved
    deduction is only defensible if the caveats that govern it travel with it.
    """
    rows = list(csv.reader((DATA / "Grading Instructions.csv").open(encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]

    general: list[str] = []
    items: list[dict] = []
    current_section = ""
    in_general = False

    for r in body:
        r = (r + [""] * len(header))[: len(header)]
        section, code_name, notes = r[0].strip(), r[1].strip(), r[2].strip()

        if section.lower().startswith("general notes"):
            in_general = True
        if in_general:
            if notes:
                general.append(notes)
            continue
        if section.lower().startswith("key:") or code_name in {"c", "p", "n"}:
            continue
        if section:
            current_section = section
        if not code_name:
            continue

        items.append({
            "code_name": code_name,
            "section": current_section,
            "notes": notes,
            "specific": {header[i]: r[i].strip() for i in (3, 4, 5) if r[i].strip()},
        })

    general_text = " ".join(general)
    out = []
    for it in items:
        specific = " ".join(f"For {k}: {v}" for k, v in it["specific"].items())
        text = (
            f"Rubric item: {it['code_name']} (proof section: {it['section']}). "
            f"Grading guidance: {it['notes']} "
            + (f"Question-specific guidance: {specific} " if specific else "")
            + f"General rules that also apply: {general_text}"
        )
        out.append(Chunk(
            chunk_id=f"RUBRIC::{it['code_name'].replace(' ', '_')}",
            doc_type="rubric_guidance",
            text=re.sub(r"\s+", " ", text).strip(),
            question_key=None,  # rubric guidance is shared across all questions
            provenance={
                "code_name": it["code_name"],
                "proof_section": it["section"],
                "has_question_specific": bool(it["specific"]),
                "source": "Grading Instructions.csv",
            },
        ))
    return out


def chunk_graded_exemplars(max_per_question: int | None = None) -> list[Chunk]:
    """
    One chunk per (submission x rubric item).

    Why not one chunk per submission? Because the retrieval question is always
    scoped to a single rubric item -- "has this student stated the inductive
    hypothesis?" -- and a whole-proof chunk would return evidence about all seven
    items at once, making it impossible to cite WHICH judgment the exemplar
    supports. Splitting per item keeps every retrieved exemplar attributable to
    exactly one rubric decision.

    The proof text is duplicated across the 7 item-chunks of a submission. That is
    deliberate: the embedded text differs (each is framed by its own rubric item
    and expert label), and provenance keeps them de-duplicable at display time.
    """
    out = []
    for csv_name, qdir, qkey in SOURCES:
        df = pd.read_csv(DATA / csv_name)
        if max_per_question:
            df = df.head(max_per_question)
        question = read_question(qdir)

        for _, row in df.iterrows():
            proof = str(row["Proof"]).strip()
            if not proof or proof.lower() == "nan":
                continue
            sub_id = f"{qkey}-{int(row['Unnamed: 0'])}"

            for col, (item_name, section) in RUBRIC_ITEMS.items():
                if col not in df.columns or pd.isna(row[col]):
                    continue
                score = int(row[col])
                text = (
                    f"Rubric item: {item_name}. Expert judgment: {LABELS[score]}. "
                    f"Claim being proved: {question} Student proof: {proof}"
                )
                out.append(Chunk(
                    chunk_id=f"{sub_id}::{col}",
                    doc_type="graded_exemplar",
                    text=re.sub(r"\s+", " ", text).strip(),
                    question_key=qkey,
                    provenance={
                        "submission_id": sub_id,
                        "question_key": qkey,
                        "rubric_item": item_name,
                        "rubric_column": col,
                        "proof_section": section,
                        "score": score,
                        "label": LABELS[score],
                        "graded_by": str(row.get("Graded.By", "")),
                        "assessment": str(row.get("Assessment", "")),
                        "semester": str(row.get("Semester", "")),
                    },
                ))
    return out


def build_index(max_per_question: int | None = None) -> list[Chunk]:
    return chunk_rubric_guidance() + chunk_graded_exemplars(max_per_question)


def access_filter(chunks: list[Chunk], question_key: str) -> list[Chunk]:
    """
    Access-aware retrieval, enforced pre-retrieval at the index level.

    Scoping rule: when grading question X, exemplars from question Y must not be
    candidates. Their expert labels were assigned against a different claim, so
    citing one as evidence would import a judgment that was never made about this
    proof. Rubric guidance (question_key=None) is intentionally shared -- it is
    the same approved rubric for all four questions.
    """
    return [c for c in chunks if c.question_key in (None, question_key)]


def verify_data() -> bool:
    """Check the corpus is present and shaped as expected before anything else runs."""
    missing = []
    for csv_name, qdir, _ in SOURCES:
        if not (DATA / csv_name).exists():
            missing.append(csv_name)
        if not (DATA / qdir / "question.html").exists():
            missing.append(f"{qdir}/question.html")
    if not (DATA / "Grading Instructions.csv").exists():
        missing.append("Grading Instructions.csv")

    if missing:
        print(f"Corpus incomplete under {DATA}\n")
        for m in missing:
            print(f"  missing: {m}")
        print("\nSee data/README.md for how to obtain it.")
        return False

    total = 0
    for csv_name, _, qkey in SOURCES:
        df = pd.read_csv(DATA / csv_name)
        present = [c for c in RUBRIC_ITEMS if c in df.columns]
        total += len(df)
        print(f"  {qkey:14s} {len(df):5,d} rows | {len(present)}/7 rubric columns "
              f"| {df['Graded.By'].nunique()} graders")
    print(f"\nCorpus OK: {total:,} submissions under {DATA}")
    return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evidence Room ingestion")
    ap.add_argument("--verify", action="store_true", help="check corpus presence and exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="max submissions per question (for fast iteration)")
    ap.add_argument("--out", default="chunks_real.jsonl", help="output JSONL path")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(0 if verify_data() else 1)

    chunks = build_index(args.limit)

    by_type: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_type.setdefault(c.doc_type, []).append(c)

    print(f"Total chunks: {len(chunks):,}\n")
    for t, cs in by_type.items():
        w = [len(c.text.split()) for c in cs]
        print(f"  {t:18s} {len(cs):6,d} chunks | words min={min(w):4d} "
              f"max={max(w):5d} mean={sum(w)/len(w):6.1f}")

    print("\n--- rubric guidance (the deduction bank) ---")
    for c in by_type["rubric_guidance"]:
        print(f"  {c.provenance['code_name']:52s} section={c.provenance['proof_section']}")

    print("\n--- label balance per rubric item (all questions) ---")
    ex = by_type["graded_exemplar"]
    tally: dict[str, dict[str, int]] = {}
    for c in ex:
        tally.setdefault(c.provenance["rubric_item"], {}).setdefault(c.provenance["label"], 0)
        tally[c.provenance["rubric_item"]][c.provenance["label"]] += 1
    for item, counts in tally.items():
        tot = sum(counts.values())
        parts = " ".join(f"{k}={v:5d}({100*v/tot:4.1f}%)" for k, v in sorted(counts.items()))
        print(f"  {item[:50]:52s} {parts}")

    print("\n--- access filter check ---")
    for qk in ("sum-formula", "divisibility"):
        vis = access_filter(chunks, qk)
        leaked = [c for c in vis if c.question_key not in (None, qk)]
        shared = [c for c in vis if c.question_key is None]
        print(f"  {qk:14s} {len(vis):6,d} visible ({len(shared)} shared rubric), {len(leaked)} leaked")

    Path(args.out).write_text(
        "\n".join(json.dumps(c.to_dict()) for c in chunks), encoding="utf-8"
    )
    print(f"\nWrote {args.out} ({len(chunks):,} chunks)")
