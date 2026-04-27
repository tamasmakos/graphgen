# GraphGen entity resolution and graph cleanup iteration plan

> For Hermes: implement this plan conservatively. Optimize for graph quality, canonical entity structure, and downstream community detection / GraphRAG usefulness. Do not overengineer. Prefer test-first changes, small commits, and one-document real-pipeline verification.

Goal: improve the quality of the ontology-guided GraphGen knowledge graph by making entity nodes more canonical, reducing discourse-artifact fragmentation, and strengthening graph-aware entity resolution without loosening relation quality controls.

Architecture: keep the current GLiNER2 + DSPy extraction backbone. Add a modest canonicalization and graph-aware resolution layer between extraction and downstream graph analysis. Treat this as a graph-cleaning / structural-quality iteration, not as a broader extraction-system rewrite.

Tech stack: Python, unittest, NetworkX, existing GraphGen pipeline modules, current diagnostics JSON outputs, arXiv-backed methodology guidance.

---

## Why this iteration is next

The current real one-segment smoke run is already operational:
- extraction processed: 3
- successful: 3
- total entities: 17
- total relations: 9
- no extraction errors

The diagnostics suggest the current bottleneck is not raw extraction viability. The bottleneck is graph cleanliness:
- entity aliases and discourse shorthand remain fragmented
- some role/title nodes behave like entities
- policy/theme concepts, named entities, and discourse artifacts are mixed too loosely
- semantic resolution currently contributes little (`merged_nodes: 0`, `clusters_found: 0`)

This directly matters for the thesis because the downstream target is not merely extraction quantity. It is a graph whose communities support interpretable topic modeling and future GraphRAG-style use.

---

## Design principles

1. Preserve the current extraction backbone unless a test proves a change is needed.
2. Prefer canonicalization and graph cleanup over extracting more low-quality triplets.
3. Use graph-based evidence in entity resolution, but keep it interpretable.
4. Keep concept nodes that help topic modeling, but distinguish them more clearly from named entities and role artifacts.
5. Make every change observable in diagnostics and measurable in the one-document real run.

---

## Intended end state

After this iteration:
- obvious entity aliases collapse more reliably
- hub nodes are cleaner and less fragmented
- role/title artifacts are less likely to pollute the final graph
- relation endpoints are more canonical and stable
- community detection input is structurally cleaner
- diagnostics show exactly what canonicalization and resolution did

Non-goals for this iteration:
- no GNNs
- no KGE revival
- no link prediction system
- no major relation-schema redesign
- no broad increase in recall by loosening acceptance criteria

---

## Task 1: Add a small canonicalization helper module

Objective: create a single place for modest, explicit, testable entity normalization rules.

Files:
- Create: `graphgen/pipeline/graph_cleaning/canonicalization.py`
- Test: `tests/test_entity_canonicalization.py`

Step 1: Write failing tests for canonicalization behavior

Test cases to cover:
- `EU` stays canonical and does not silently become `EUROPE`
- `European Central Bank` and `EUROPEAN_CENTRAL_BANK` normalize compatibly
- role-only strings like `PRIME_MINISTER` are detectable as role/title artifacts
- slogans like `WHATEVER_IT_TAKES` are not treated as named persons/organizations
- simple punctuation/case variants normalize consistently

Expected API sketch:
```python
from graphgen.pipeline.graph_cleaning.canonicalization import (
    normalize_surface_form,
    classify_surface_form,
    are_potential_aliases,
)
```

Step 2: Run the new test file and verify failure

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_canonicalization -v`

Expected: FAIL because the module does not yet exist.

Step 3: Implement minimal canonicalization helpers

Implement small helpers such as:
- `normalize_surface_form(text: str) -> str`
- `classify_surface_form(text: str) -> str` returning values like:
  - `named_entity`
  - `role_artifact`
  - `concept_like`
  - `unknown`
- `are_potential_aliases(a: str, b: str) -> bool`

Rules should be conservative:
- normalize punctuation, spacing, underscores, case
- detect role/title-only forms with a small explicit role lexicon
- do not conflate geopolitical shorthand automatically unless rule is explicit and tested

Step 4: Re-run tests to verify pass

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_canonicalization -v`

Expected: PASS

Step 5: Commit

```bash
git add -f tests/test_entity_canonicalization.py graphgen/pipeline/graph_cleaning/canonicalization.py
git commit -m "feat: add conservative entity canonicalization helpers"
```

---

## Task 2: Make graph resolution type-aware and alias-aware

Objective: improve `BlockingResolver` so it uses better lexical and type gating before merges.

Files:
- Modify: `graphgen/pipeline/graph_cleaning/resolution.py`
- Test: `tests/test_entity_resolution_graph_cleanup.py`

Step 1: Write failing tests for graph resolution candidate filtering

Create tests asserting:
- obvious aliases can merge:
  - `EUROPEAN_CENTRAL_BANK` vs `European Central Bank`
- role-only node should not merge into a person just because of weak lexical overlap:
  - `PRIME_MINISTER` should not merge into `MARIO_DRAGHI`
- concept nodes should not merge with named entities:
  - `MIGRATION` should not merge with `ITALY`
- same-type, high-overlap, neighborhood-compatible nodes are merge candidates

Step 2: Run tests and confirm failure

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_resolution_graph_cleanup -v`

Expected: FAIL before implementation.

Step 3: Implement minimal resolver improvements

In `resolution.py`, add:
- type compatibility gate
- alias-aware lexical match using canonicalized surface forms
- role/concept artifact penalty
- optional neighborhood compatibility score using 1-hop relation labels / neighbors

Keep the scoring simple and interpretable.

Preferred scoring idea:
- lexical score
- type compatibility bonus / mismatch penalty
- neighborhood overlap bonus
- role-artifact penalty

Do not introduce external dependencies.

Step 4: Re-run tests to verify pass

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_resolution_graph_cleanup -v`

Expected: PASS

Step 5: Commit

```bash
git add -f tests/test_entity_resolution_graph_cleanup.py graphgen/pipeline/graph_cleaning/resolution.py
git commit -m "fix: make entity resolution alias-aware and type-aware"
```

---

## Task 3: Add lightweight node-role classification into diagnostics

Objective: make diagnostics explicitly show which extracted nodes are being treated as named entities, concepts, or role/title artifacts.

Files:
- Modify: `graphgen/pipeline/entity_relation/extraction.py`
- Modify: `graphgen/pipeline/graph_cleaning/resolution.py`
- Test: `tests/test_pipeline_regressions.py`

Step 1: Write a failing regression test

Add a test asserting chunk or segment diagnostics include classification metadata such as:
- `surface_form_class`
- `canonical_form`
- or an equivalent minimal field set

Use a tiny fake chunk with entities such as:
- `MARIO_DRAGHI`
- `PRIME_MINISTER`
- `MIGRATION`

Step 2: Run the targeted regression test and verify failure

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_pipeline_regressions.PipelineRobustnessRegressionTests -v`

Expected: FAIL for the new assertion.

Step 3: Implement the minimal diagnostic enrichment

Diagnostics should expose:
- original text
- canonical form
- surface-form class
- whether it was eligible for merge consideration

Do not add excessive nesting.

Step 4: Re-run the targeted regression and broad suite

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_pipeline_regressions -v`

Then:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_pipeline_regressions tests.test_gliner2_pipeline_integration tests.test_gliner2_prototype tests.test_local_smoke -v`

Expected: PASS

Step 5: Commit

```bash
git add -f tests/test_pipeline_regressions.py graphgen/pipeline/entity_relation/extraction.py graphgen/pipeline/graph_cleaning/resolution.py
git commit -m "feat: expose canonical entity classes in diagnostics"
```

---

## Task 4: Add a one-document resolution evaluation helper

Objective: measure whether graph cleanup actually improves the graph for downstream topic/community use.

Files:
- Create: `graphgen/evaluation/entity_resolution_eval.py`
- Test: `tests/test_entity_resolution_eval.py`

Step 1: Write failing tests for evaluation metrics

Test metrics:
- unique entity count before/after merge
- merged node count
- top hub list before/after
- type distribution before/after
- optional alias cluster summary

Step 2: Run tests and verify failure

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_resolution_eval -v`

Expected: FAIL because helper does not yet exist.

Step 3: Implement a minimal evaluator

The helper should accept a graph and return a dict with fields like:
- `entity_nodes_before`
- `entity_nodes_after`
- `merged_nodes`
- `top_degree_nodes_before`
- `top_degree_nodes_after`
- `surface_form_class_counts`

This is for structural evaluation, not benchmark publication.

Step 4: Re-run tests and verify pass

Run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_resolution_eval -v`

Expected: PASS

Step 5: Commit

```bash
git add -f tests/test_entity_resolution_eval.py graphgen/evaluation/entity_resolution_eval.py
git commit -m "feat: add entity resolution structural evaluation helper"
```

---

## Task 5: Run the real one-segment pipeline again and inspect outputs

Objective: verify that graph cleanup improves structure without damaging relation quality.

Files:
- Use existing config: `/tmp/graphgen_full_smoke_diag.yaml`
- Inspect existing output dir or a new temporary output dir

Step 1: Ensure diagnostics are enabled in the one-segment config

Expected config characteristics:
- real pipeline
- `ner_backend: gliner2`
- `relation_backend: dspy`
- diagnostics enabled
- analytics disabled
- iterative disabled

Step 2: Run the real pipeline

If a valid Groq key is available in the environment, run:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m graphgen.main`

Or use the direct smoke configuration path already established in the repo workflow.

Step 3: Inspect artifacts

Inspect:
- `entity_resolution_report.json`
- `knowledge_graph.graphml`
- per-chunk diagnostics
- segment diagnostics

Specifically compare:
- merged node count
- whether `PRIME_MINISTER` remains a polluting standalone node
- whether obvious aliases are merged better
- whether accepted relations remain stable
- whether major hubs become cleaner

Step 4: Record a concise structural evaluation note

Create a short markdown note or PR comment covering:
- before vs after merged nodes
- before vs after hub cleanliness
- relation count changes
- whether community/topic-readiness improved

Step 5: Commit only code changes, not generated outputs

Generated outputs should remain untracked unless explicitly requested.

---

## Task 6: Optional final refinement — role nodes policy

Objective: decide whether role/title-only nodes should remain nodes or become attributes in the final graph.

Files:
- Likely modify: `graphgen/pipeline/entity_relation/extraction.py`
- Possibly modify: `graphgen/pipeline/graph_cleaning/resolution.py`
- Test: `tests/test_pipeline_regressions.py`

Only do this after Tasks 1–5.

Decision rule:
- if a role/title behaves mostly as noise, demote it from node status
- if it participates in stable, repeated, semantically useful relations across the corpus, keep it as a typed node

For now, prefer the smallest change possible.

---

## Evaluation criteria

This iteration is successful if:
- obvious alias merges improve
- role/title artifacts are reduced or better classified
- named-entity hubs become cleaner
- no regression in the current high-quality factual relations
- broad unittest suite remains green
- one-segment diagnostics become easier to justify academically

This iteration is NOT judged by maximizing:
- raw relation count
- total node count
- complexity of the resolution model

---

## Literature rationale to cite in notes / thesis-facing discussion

Use these claims conservatively:
- GLiNER supports flexible label-guided NER, which justifies the ontology-guided candidate-label approach already used.
- GraphRAG motivates improving the entity graph as a structured index for corpus-level sensemaking.
- LightRAG reinforces the usefulness of combining graph structure with retrieval-oriented representations, but does not require copying its architecture.
- EAGER suggests that embedding-assisted entity resolution works best when embeddings are combined with attribute/value signals rather than used alone.

Thesis-facing interpretation:
- the next improvement is not a broader extractor, but a cleaner structural graph
- this better matches the thesis goal of topic modeling via graph communities and GraphRAG-compatible summaries

---

## Final verification commands

Targeted tests:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_entity_canonicalization tests.test_entity_resolution_graph_cleanup tests.test_entity_resolution_eval -v`

Regression suite:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_pipeline_regressions -v`

Broad suite:
`PYTHONPATH=/root/graphgen /root/graphgen/.venv/bin/python -m unittest tests.test_pipeline_regressions tests.test_gliner2_pipeline_integration tests.test_gliner2_prototype tests.test_local_smoke tests.test_entity_canonicalization tests.test_entity_resolution_graph_cleanup tests.test_entity_resolution_eval -v`

Real one-document verification:
- run the established diagnostic-enabled one-segment real pipeline
- inspect JSON diagnostics and GraphML
- confirm structural quality improved without reintroducing junk relations

---

## Suggested commit sequence

1. `feat: add conservative entity canonicalization helpers`
2. `fix: make entity resolution alias-aware and type-aware`
3. `feat: expose canonical entity classes in diagnostics`
4. `feat: add entity resolution structural evaluation helper`

---

## Notes for Hermes

- Keep the implementation modest and explicit.
- Do not introduce large ontology matching machinery.
- Do not broaden relation acceptance criteria as part of this iteration.
- Prefer graph-quality improvements that directly support Leiden communities and future GraphRAG use.
- Push commits to the existing PR branch when each chunk of work is verified.