# Automated label-space design and testing plan

> For Hermes: planning only. Do not implement from this document without a separate execution step.

Goal

Remove reliance on manual domain hints/descriptions in GraphGen and replace them with an automated, corpus-aware label-space pipeline that works across political text, fiction, and other domains.

Why the current hints/descriptions exist

Today the manual hints are compensating for three real problems:

1. GLiNER2 works better with a compact candidate schema than with a huge flat label list.
2. Short label names are ambiguous without descriptions.
3. Candidate selection needs some signal to decide which labels are relevant for a given chunk.

So the hardcoded values are not random; they are doing real work. The problem is where that work lives: inside source code, with domain-specific assumptions. That is brittle and not acceptable for a flexible system.

Design principle

Keep the functions, remove the hand-curated domain knowledge.

In other words:
- keep schema narrowing
- keep label descriptions
- keep candidate selection
- replace manual domain hints with automated metadata derived from ontology and/or corpus evidence

Architecture summary

Replace the current manual-hint path with a 3-stage automated label-space pipeline:

1. Label metadata generation
2. Candidate label retrieval per chunk
3. Optional corpus bootstrap for label-poor settings

The output of this pipeline is the same kind of object the extractor needs today:
- candidate labels
- descriptions
- supporting metadata/diagnostics

But it is generated automatically per run, not hardcoded in code.

Proposed runtime design

A. Label metadata generation

Input sources, in priority order:
1. Ontology metadata, if available
   - rdfs:label
   - skos:prefLabel / altLabel if present
   - rdfs:comment / definition-like fields
   - parent/child relations
2. Config-provided labels, if ontology is absent
3. Corpus bootstrap labels, if both are weak or absent

For each label, build a LabelProfile:
- canonical_label
- aliases derived automatically from label text and ontology metadata
- description derived automatically
- parent_labels
- child_labels
- embedding vector of label profile text
- provenance: ontology | config | bootstrap

Automatic alias generation rules
- split snake_case / camelCase / URI local names
- lowercase lexical variants
- singular/plural normalization where cheap
- ontology alt labels if present
- no domain-specific token lists in code

Automatic description generation rules
- preferred: ontology definition/comment text
- fallback: templated description from graph structure, e.g.
  - "Label X with parent Y and children A, B, C"
- optional later enhancement: LLM-generated description from ontology context, cached per run

B. Candidate label retrieval per chunk

For each chunk:
1. compute chunk embedding
2. score labels using a hybrid retriever:
   - semantic similarity between chunk embedding and label-profile embedding
   - lexical overlap with aliases/descriptions using BM25 or simple token scoring
   - optional graph prior such as label prevalence in bootstrap sample
3. choose top-k labels
4. build GLiNER2 schema from those labels and auto-generated descriptions

Important: this preserves the useful "narrowed schema" behavior without any hardcoded hints like political_hints, TOP_LEVEL_ALIASES, or handcrafted descriptions.

C. Corpus bootstrap mode for label-poor or no-ontology texts

This is the path that makes fiction/general prose flexible.

Bootstrap pass on a small sample of chunks:
1. run a generic mention detector
   - first option: generic GLiNER model with very broad labels
   - fallback: spaCy NER + noun phrase extraction
2. collect mention spans plus local context windows
3. embed mention-context pairs
4. cluster them
5. induce provisional label names/descriptions automatically
   - deterministic version: cluster summary from most informative terms
   - optional higher-quality version: one LLM pass per cluster, cached
6. feed these induced labels into the main extraction pass

This means Harry Potter should not require code edits. The system should discover labels like character/school/place/object/event from the corpus itself.

Recommended config model

Add config-driven strategies instead of code-level assumptions:

extraction:
  label_source: hybrid              # ontology | config | bootstrap | hybrid
  label_metadata_strategy: hybrid   # ontology | templated | llm | hybrid
  candidate_selection_strategy: hybrid  # embedding | lexical | hybrid
  candidate_top_k: 5
  segmentation_mode: line           # already a useful generalization
  min_segment_words: 0
  bootstrap:
    enabled: true
    sample_chunks: 32
    max_induced_labels: 12
    clustering_method: hdbscan
    cache_file: thesis_outputs/label_space_manifest.json

Diagnostics to persist:
- label_space_manifest.json
- per-chunk candidate retrieval diagnostics
- label provenance counts
- bootstrap cluster summaries

What to delete / deprecate

These should stop driving runtime behavior:
- graphgen/prototype_gliner2_ontology.py manual top-level group rules
- hardcoded domain alias lists
- hardcoded political/fantasy term hints
- hardcoded default descriptions for domain selection logic

A minimal generic fallback description template is fine.
A domain-specific hardcoded vocabulary is not.

Concrete implementation plan

Phase 0: reset the wrong direction

Objective
- discard any local experiments that inject domain-specific tokens into production code
- keep only the generic segmentation work if desired

Files to inspect/reset
- graphgen/prototype_gliner2_ontology.py
- graphgen/pipeline/entity_relation/extraction.py
- graphgen/config/settings.py
- tests/test_pipeline_regressions.py

Verification
- production code contains no Harry Potter- or political-specific retrieval hints

Phase 1: isolate label-space logic behind a neutral interface

Objective
- create a dedicated label-space module so extraction no longer depends on prototype_gliner2_ontology internals

Create
- graphgen/pipeline/entity_relation/label_space.py

Public API proposal
- build_label_profiles(labels, ontology_metadata=None, config=None) -> list[LabelProfile]
- select_candidate_labels_for_text(text, label_profiles, top_k, strategy="hybrid") -> CandidateSelectionResult
- build_schema_from_candidate_profiles(candidate_profiles) -> dict

Data types
- LabelProfile
- CandidateSelectionResult

Modify
- graphgen/pipeline/entity_relation/extraction.py

Verification
- extraction path calls label_space.py only
- no hardcoded manual hint tables are consulted

Phase 2: ontology-driven automatic metadata

Objective
- derive aliases/descriptions from ontology instead of code constants

Modify
- graphgen/utils/ontology_parser.py
- graphgen/utils/labels.py
- graphgen/pipeline/entity_relation/label_space.py

Add support for:
- canonical label text
- alt labels
- comments/definitions
- parent/child edges

Verification
- for ontology-enabled runs, label profiles are populated from ontology files only
- manifest records provenance=ontology

Phase 3: corpus bootstrap for label-poor corpora

Objective
- make non-political and non-ontology corpora workable without code edits

Create
- graphgen/pipeline/entity_relation/label_bootstrap.py

Responsibilities
- sample chunks
- generic mention proposal
- mention-context embedding
- clustering
- induced label summaries
- cache results into output artifacts

Modify
- graphgen/orchestrator.py or graphgen/pipeline entrypoint that prepares extraction config
- graphgen/config/settings.py for bootstrap settings

Verification
- with ontology disabled and no manual hint tables, the pipeline still induces a usable label space on a small fiction sample

Phase 4: retrieval and schema diagnostics

Objective
- make candidate selection inspectable and debuggable

Create or extend output artifacts
- output_dir/thesis_outputs/label_space_manifest.json
- diagnostics/chunk_candidate_labels_<chunk_id>.json

Each diagnostic should contain
- retrieved candidates
- scores by strategy component
- label provenance
- schema text passed to GLiNER2

Verification
- if label selection is bad, we can explain why without reading code

Testing framework

Tier 1: fast unit tests

Create
- tests/test_label_space.py
- tests/test_label_bootstrap.py

Unit test targets

1. Label profile generation
- aliases come from label text and ontology metadata, not hardcoded tables
- descriptions come from ontology comments or templates
- provenance is recorded correctly

2. Candidate retrieval
- lexical-only cases work
- embedding-only cases work
- hybrid ranking behaves sensibly

3. Schema generation
- selected candidate labels are converted into the expected GLiNER2 schema shape
- no manual domain descriptions are required

4. Bootstrap label induction
- given synthetic mention/context samples, clustering produces stable induced labels
- caching and reload path work

Tier 2: contract tests against hardcoding regressions

Create
- tests/test_no_manual_domain_hints.py

Purpose
- fail if production modules reintroduce domain-specific token lists in runtime candidate-selection code

Suggested rule
- scan production files for forbidden runtime markers such as:
  - hardcoded corpus-specific names
  - manually curated alias arrays for specific domains
- allow such strings only in:
  - tests
  - fixtures
  - documentation

Files in scope
- graphgen/pipeline/entity_relation/*.py
- graphgen/utils/*.py

Tier 3: fixture-based corpus tests

Create fixture directories
- tests/fixtures/corpora/political/
- tests/fixtures/corpora/fiction/
- tests/fixtures/corpora/generic/

Use tiny corpora, for example:
- political: 20-40 short transcript lines
- fiction: 2-3 short Harry Potter-style paragraphs or public-domain fiction substitute if licensing matters
- generic: business/news style paragraphs

Expected assertions should be behavioral, not exact-string brittle:
- label space non-empty
- at least N retrieved candidate labels per chunk
- extracted entities/relations non-zero
- no manual-hint path invoked
- diagnostics written

Tier 4: bounded end-to-end smoke tests

Create
- tests/test_label_space_e2e.py

Smoke scenarios
1. ontology-enabled political sample
2. ontology-disabled fiction sample using bootstrap
3. config-label-only generic sample

Assertions
- run completes
- graph_schema.json exists
- knowledge_graph.graphml exists
- label_space_manifest.json exists
- candidate diagnostics exist
- zero reliance on manual domain hint tables

Tier 5: quality benchmarks

Not CI-blocking at first, but tracked locally.

Create benchmark script
- scripts/benchmark_label_space.py

Measure per corpus:
- extraction completion rate
- entities extracted
- relations extracted
- candidate-label entropy / diversity
- runtime overhead of bootstrap
- topic/community counts downstream

Acceptance criteria for replacement

Must-have
- no production code contains domain-specific hint vocabularies for candidate selection
- ontology-enabled runs derive descriptions/aliases automatically
- ontology-disabled runs can bootstrap a usable label space from corpus samples
- fiction and political smoke fixtures both pass without code edits
- diagnostics explain candidate-label choices

Nice-to-have
- bootstrap results cached and reusable across reruns
- optional LLM enrichment of label descriptions behind config flag
- benchmark script compares old vs new strategy on bounded corpora

Likely files to change in implementation

Create
- graphgen/pipeline/entity_relation/label_space.py
- graphgen/pipeline/entity_relation/label_bootstrap.py
- tests/test_label_space.py
- tests/test_label_bootstrap.py
- tests/test_no_manual_domain_hints.py
- tests/test_label_space_e2e.py
- tests/fixtures/corpora/political/*
- tests/fixtures/corpora/fiction/*
- tests/fixtures/corpora/generic/*
- scripts/benchmark_label_space.py

Modify
- graphgen/pipeline/entity_relation/extraction.py
- graphgen/utils/ontology_parser.py
- graphgen/utils/labels.py
- graphgen/config/settings.py
- graphgen/orchestrator.py

Migration strategy

Step 1
- add new modules behind config flags

Step 2
- keep old manual path temporarily as legacy fallback

Step 3
- add tests proving new path works across domains

Step 4
- switch default to automated path

Step 5
- remove legacy manual-hint code once green on all fixture suites

Risks and tradeoffs

1. Bootstrap adds runtime cost
- mitigate with sample-chunk cap and cache

2. Fully automatic label induction may be noisier than curated hints
- mitigate with diagnostics, hybrid retrieval, and benchmark gates

3. Ontology metadata quality may be poor or absent
- mitigate with templated fallback plus bootstrap mode

4. Exact induced label names may vary
- tests should validate behavior and provenance, not brittle exact wording

Recommended first implementation slice

Do not try to solve everything at once.

First slice:
1. implement neutral label profiles from ontology/config labels
2. implement hybrid candidate retrieval with automatic descriptions
3. add fixture tests for political + fiction small samples
4. only then add bootstrap induction for label-poor corpora

This gets us off manual hints quickly without overengineering the first pass.

Suggested immediate next step

Before any new code:
- reset the current experimental hardcoded hint edits
- keep the useful segmentation generalization idea separate
- implement the new neutral label-space module behind a config flag

Short answer to the original question

We need hints/descriptions because the extractor needs label narrowing and label semantics.
We do not need those hints to be hand-written in code.
They should be generated automatically from ontology metadata and, when needed, from a lightweight corpus bootstrap pass.
