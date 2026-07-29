# Opportunity Brief

**Deduction-Bank Grading for Proof-Based Math Assignments**

| | |
|---|---|
| **Workflow** | Grading of deterministic, proof-based problem sets (e.g. Discrete Structures induction proofs) by Faculty Interns (FIs) at Ashesi University. |
| **Operator** | Faculty Intern grading a cohort's assignments and quizzes — author has direct, first-hand experience in this role. |
| **Prepared by** | Daniel Byiringiro — Pareto FDE Academy, Cohort 01 |

## The Problem

Each week, a Faculty Intern grades ~240 scripts (one quiz + one assignment), or ~1,000/month, across 118 students. Per assignment: lecturer creates it → FI drafts a rubric → lecturer reviews and corrects it → FI grades, authoring a fresh judgment each time a recurring error appears and recording it as a deduction tag (e.g. "–0.5 if P(k) not stated").

Two costs recur every assignment: authoring the same judgment repeatedly even after an error type has already been seen this session, and no guarantee a tag is applied consistently once it exists — application depends on the FI remembering it, not on the tag being checked against each remaining script.

## Baseline & Target

| | |
|---|---|
| **Baseline** | ~3–4 min/script once rubric is mastered — ~58 hrs/month per FI at full volume. No current tracking of within-session consistency. |
| **Target metric** | **Primary — time/script:** reduce it by shifting the FI's action from authoring a deduction to reviewing a suggested one.<br>**Secondary — consistency:** once a tag exists, surface it on every remaining eligible script, not only ones graded after the FI happened to remember it. |

## Why Now

Author has personally performed this exact workflow — deduction patterns, failure modes, and real per-script timing are already known firsthand, not second-hand.

## First Deployment

A lightweight companion tool: as the FI creates a deduction tag, it's added to a session bank; on later scripts, matching tags are surfaced as suggestions for the FI to mark applicable or not — never auto-applied. Grading shifts from authoring judgments to reviewing them. Forward-only by design: scripts graded before a tag existed are not retroactively re-checked — a named limitation, not a gap. Scoped to one FI, one assignment cycle — deployable within the sprint, testable against both target metrics.

## Risk Register

| Risk | Detail |
|---|---|
| **Data** | Student scripts are sensitive (FERPA-equivalent). No public LLM logging; access restricted to authorized grading staff. |
| **Permissions** | Rubrics/banks are owned per-FI, per-cohort — not shared, since lecturers may approve different rubrics for the same assignment. |
| **Consistency** | Primary risk is drift within one FI's own session — not cross-grader standardization. |
| **Hallucination** | Misapplying a correct deduction to a step the student actually completed (just phrased differently) outweighs the risk of inventing a wrong one. |
| **Adoption** | Every applied deduction must trace back to the approved rubric — a black-box score is worse than the status quo. |
| **Scope** | Forward-only tagging leaves early scripts unre-checked against later-discovered tags — acceptable for v1, but should be named to reviewers, not discovered by them. |
