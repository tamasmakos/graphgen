# Experiment Log & Decision Record

This document tracks key architectural decisions, pivots, and experimental hypotheses in the graph-based topic modeling project.

## 2026-04-27: Non-iterative local scaling path validated

### Decision
The active thesis workflow is now the non-iterative `KnowledgePipeline`, not the earlier iteration-as-evaluation setup.

### Rationale
Recent local validation showed that the earlier CPU-bound GLiNER path was not a practical route for thesis-scale experimentation on the available hardware. In contrast, the GLiNER2 + DSPy + Node2Vec configuration completed reliably while preserving the graph-construction and summarization stages required by the thesis.

### Validated local path
- **Runtime**: non-iterative `KnowledgePipeline`
- **NER backend**: `gliner2`
- **Relation extraction**: `dspy`
- **Community weighting**: Node2Vec enabled
- **Execution constraints**: CPU-only, `max_concurrent_chunks = 1`, chunk-budgeted local runs

### Scaling evidence
Completed local runs produced the following progression:
- **48 chunks** -> 531 nodes / 1341 edges / 10 topics / 39 subtopics
- **64 chunks** -> 661 nodes / 1733 edges / 14 topics / 47 subtopics
- **96 chunks** -> 923 nodes / 2529 edges / 16 topics / 66 subtopics
- **128 chunks** -> 1155 nodes / 3239 edges / 14 topics / 72 subtopics
- **160 chunks** -> 1377 nodes / 3946 edges / 14 topics / 80 subtopics
- **187 chunks** -> 1617 nodes / 4670 edges / 17 topics / 91 subtopics

### Interpretation
These runs validate operational scalability of the non-iterative pipeline on modest local hardware. They show stable graph growth, successful hierarchical summarization, and the continued production of thesis-facing artifacts such as GraphML, entity-resolution reports, provenance manifests, and topic-separation reports.

### Caution
The available topic-separation reports remain statistically inconclusive and repeatedly state "Insufficient data for analysis." Accordingly, the present evidence supports feasibility, graph growth, and hierarchical summary generation more strongly than strong inferential claims about semantic topic separation.

### Implication for thesis framing
The earlier iterative outputs remain part of the project's historical record, but they should no longer be treated as the mainline implementation narrative. Future work on graph growth should be framed in terms of principled incremental accumulation, entity resolution, and merge eligibility rather than iteration count alone.
