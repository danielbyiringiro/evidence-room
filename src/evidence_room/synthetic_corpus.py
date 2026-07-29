"""
Synthetic corpus for the Evidence Room lab.

Three document types, deliberately messy in the ways real grading data is messy:
  - rubrics        : approved point allocations per proof step
  - deduction_tags : the FI's growing bank, written in shorthand
  - scripts        : student proof attempts, including correct, partial, and wrong

All data is synthetic. No real student work.
"""

RUBRICS = [
    {
        "rubric_id": "R-DS-A3-Q1",
        "cohort": "cohort-a",
        "assignment": "A3",
        "question": "Prove that n^3 + 2n is divisible by 3 for all positive integers n.",
        "total_points": 6,
        "criteria": [
            {"criterion_id": "C1", "points": 1,
             "expected": "Base case n=1 evaluated explicitly and shown divisible by 3."},
            {"criterion_id": "C2", "points": 1,
             "expected": "Inductive hypothesis stated: assume P(k) holds, i.e. 3 | k^3 + 2k."},
            {"criterion_id": "C3", "points": 2,
             "expected": "Expansion of (k+1)^3 + 2(k+1) carried out correctly."},
            {"criterion_id": "C4", "points": 1,
             "expected": "Expression regrouped to show it equals (k^3+2k) + 3(k^2+k+1)."},
            {"criterion_id": "C5", "points": 1,
             "expected": "Conclusion drawn: P(k+1) holds, therefore statement true for all n."},
        ],
    },
    {
        "rubric_id": "R-DS-A3-Q2",
        "cohort": "cohort-b",
        "assignment": "A3",
        "question": "Prove that the sum of the first n odd integers equals n^2.",
        "total_points": 6,
        "criteria": [
            {"criterion_id": "C1", "points": 1,
             "expected": "Base case n=1 shown: sum is 1, equals 1^2."},
            {"criterion_id": "C2", "points": 1,
             "expected": "Inductive hypothesis stated: assume sum of first k odds = k^2."},
            {"criterion_id": "C3", "points": 2,
             "expected": "Adds the (k+1)th odd integer, 2k+1, to both sides correctly."},
            {"criterion_id": "C4", "points": 1,
             "expected": "Simplifies k^2 + 2k + 1 to (k+1)^2."},
            {"criterion_id": "C5", "points": 1,
             "expected": "Conclusion drawn for all n."},
        ],
    },
]

# The FI's bank, written the way an FI actually writes it: terse, inconsistent
# capitalisation, sometimes referencing the criterion, sometimes not.
DEDUCTION_TAGS = [
    {"tag_id": "T01", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C2", "delta": -0.5,
     "text": "state P(k) first", "note": "hypothesis never explicitly written"},
    {"tag_id": "T02", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C4", "delta": -0.5,
     "text": "specify that it is 3 times an integer",
     "note": "wrote 3(k^2+k+1) but never said this is divisible by 3"},
    {"tag_id": "T03", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C4", "delta": -0.5,
     "text": "specify that the result is divisible by 3", "note": "regrouping done, claim not made"},
    {"tag_id": "T04", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C5", "delta": -0.5,
     "text": "conclusion", "note": "no closing statement for all n"},
    {"tag_id": "T05", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C3", "delta": -1.0,
     "text": "expansion error", "note": "(k+1)^3 expanded incorrectly"},
    {"tag_id": "T06", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C1", "delta": -1.0,
     "text": "no base case", "note": "jumped straight to inductive step"},
    {"tag_id": "T07", "rubric_id": "R-DS-A3-Q1", "criterion_id": "C2", "delta": -0.5,
     "text": "assumed what was to be proven",
     "note": "circular: used P(k+1) inside the proof of P(k+1)"},
    {"tag_id": "T08", "rubric_id": "R-DS-A3-Q2", "criterion_id": "C3", "delta": -0.5,
     "text": "wrong (k+1)th odd term", "note": "used 2k-1 instead of 2k+1"},
    {"tag_id": "T09", "rubric_id": "R-DS-A3-Q2", "criterion_id": "C4", "delta": -0.5,
     "text": "did not factor to (k+1)^2", "note": "left as k^2+2k+1"},
    {"tag_id": "T10", "rubric_id": "R-DS-A3-Q2", "criterion_id": "C5", "delta": -0.5,
     "text": "conclusion", "note": "no closing statement"},
]

# Student attempts. proof_steps are the natural chunking unit.
SCRIPTS = [
    {
        "script_id": "S001", "cohort": "cohort-a", "rubric_id": "R-DS-A3-Q1",
        "proof_steps": [
            "For n = 1: 1^3 + 2(1) = 3, and 3 is divisible by 3. So the base case holds.",
            "Assume the statement is true for n = k. That is, 3 divides k^3 + 2k.",
            "Now (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2.",
            "This equals (k^3 + 2k) + 3k^2 + 3k + 3 = (k^3 + 2k) + 3(k^2 + k + 1).",
            "Both terms are divisible by 3, so P(k+1) holds. By induction the statement is true for all positive integers n.",
        ],
    },
    {
        "script_id": "S002", "cohort": "cohort-a", "rubric_id": "R-DS-A3-Q1",
        "proof_steps": [
            "For n = 1 we get 3 which works.",
            "Now (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2.",
            "= (k^3 + 2k) + 3(k^2 + k + 1).",
            "So it works.",
        ],
    },
    {
        "script_id": "S003", "cohort": "cohort-a", "rubric_id": "R-DS-A3-Q1",
        "proof_steps": [
            "Assume 3 | k^3 + 2k.",
            "(k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 2k + 2.",
            "= (k^3 + 2k) + 3(k^2 + k) + 2.",
            "Hence divisible by 3 for all n.",
        ],
    },
    {
        "script_id": "S004", "cohort": "cohort-b", "rubric_id": "R-DS-A3-Q2",
        "proof_steps": [
            "n = 1: the first odd integer is 1 and 1^2 = 1. Base case holds.",
            "Assume 1 + 3 + ... + (2k-1) = k^2.",
            "Adding the next odd number: k^2 + (2k+1).",
            "k^2 + 2k + 1 = (k+1)^2, which is what we wanted.",
            "Therefore by induction the result holds for all n >= 1.",
        ],
    },
    {
        "script_id": "S005", "cohort": "cohort-b", "rubric_id": "R-DS-A3-Q2",
        "proof_steps": [
            "Base: n=1 gives 1 = 1^2. OK.",
            "Assume true for k.",
            "Then adding 2k-1 gives k^2 + 2k - 1.",
            "So the formula holds.",
        ],
    },
]
