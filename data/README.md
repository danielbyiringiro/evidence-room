# Data

**Nothing in this directory is tracked by git**, apart from this file.

The corpus is real student coursework collected under institutional consent.
It is not ours to redistribute, so the pipeline ships without it and you supply
it locally.

## Obtaining the corpus

The written-induction dataset is published via Dataverse. Download the archive
and unpack it here so the tree looks like:

```
data/
├── Grading Instructions.csv
├── online_consent.pdf
├── WrittenInduction_v1-all.csv
├── WrittenInduction_v1/
│   ├── info.json
│   └── question.html
├── WrittenInduction_Sum_v3-sp23.csv
├── WrittenInduction_Sum_v3/
├── WrittenInduction_Recurrence_v3-all.csv
├── WrittenInduction_Recurrence_v3/
├── WrittenInduction_Divisibilty_v4-sp23.csv
└── WrittenInduction_Divisibilty_v4/
```

Then verify:

```bash
python -m evidence_room.ingest --verify
```

## What is in it

| | |
|---|---|
| Submissions | ~3,600 graded student proofs |
| Questions | 4 induction problems (sum formula, factorial sum, recurrence, divisibility) |
| Rubric | 7 items, scored 0/1/2 (not present / partial / correct) |
| Graders | 2-4 named course staff per question set |
| Guidance | `Grading Instructions.csv` -- written rationale per rubric item |

## Handling rules

- Do not commit any file from this directory.
- Do not paste proof text into issues, PRs, slides, or demo screenshots.
- Grader names appear in a `Graded.By` column. Aggregate them; do not single
  anyone out in published work.
- Derived artifacts (chunk files, embedding indexes) contain verbatim student
  text and are gitignored for the same reason as the source.
