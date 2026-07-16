"""Semantic chunk-to-ontology matching.

Loads every RDF/OWL file in the ontology directory once, builds an
in-memory registry of ``(description, labels, embedding)`` per ontology,
and matches each incoming text chunk to its most semantically similar
ontology by cosine similarity of sentence embeddings.  The matched
ontology's label set then constrains entity/relation extraction so the
LLM only emits types from the ontology the chunk actually belongs to.

Everything is held in memory; embeddings are computed once at build time
and compared on the fly per chunk.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OntologyEntry:
    """A single ontology: its human description, class labels and embedding."""

    name: str
    description: str
    labels: List[str]
    n_classes: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def match_text(self) -> str:
        """Text used to represent this ontology in embedding space."""
        parts = [self.name.strip()]
        if self.description:
            parts.append(self.description.strip())
        # Fold a few labels in so ontologies with terse descriptions still
        # carry domain signal.
        if self.labels:
            parts.append("Concepts: " + ", ".join(self.labels[:25]))
        return ". ".join(p for p in parts if p)


class OntologyRegistry:
    """In-memory registry of ontologies with semantic chunk matching."""

    def __init__(
        self,
        ontology_dir: str,
        embed_model_name: str = "all-MiniLM-L6-v2",
        top_level_only: bool = True,
        min_subclasses: int = 0,
        include_local_names: bool = True,
        max_labels: int = 60,
        match_threshold: float = 0.15,
    ):
        self.ontology_dir = Path(ontology_dir)
        self.embed_model_name = embed_model_name
        self.top_level_only = top_level_only
        self.min_subclasses = min_subclasses
        self.include_local_names = include_local_names
        self.max_labels = max_labels
        self.match_threshold = match_threshold

        self.entries: List[OntologyEntry] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._built = False

    def build(self) -> "OntologyRegistry":
        """Parse every ontology file and pre-compute its embedding (once)."""
        if self._built:
            return self

        from graphgen.utils.ontology_parser import extract_ontology_file_metadata

        if not self.ontology_dir.exists():
            logger.warning("Ontology directory not found: %s", self.ontology_dir)
            self._built = True
            return self

        for rdf_file in sorted(self.ontology_dir.glob("*.rdf")):
            meta = extract_ontology_file_metadata(
                str(rdf_file),
                top_level_only=self.top_level_only,
                min_subclasses=self.min_subclasses,
                include_local_names=self.include_local_names,
                max_labels=self.max_labels,
            )
            if not meta:
                logger.debug("Skipping ontology %s (no classes).", rdf_file.name)
                continue
            self.entries.append(
                OntologyEntry(
                    name=meta["name"],
                    description=meta["description"],
                    labels=meta["labels"],
                    n_classes=meta["n_classes"],
                )
            )

        if not self.entries:
            logger.warning("No usable ontologies found in %s", self.ontology_dir)
            self._built = True
            return self

        from graphgen.pipeline.embeddings.rag import get_embedding_model

        self._model = get_embedding_model(self.embed_model_name)
        if self._model is None:
            logger.warning("Embedding model unavailable; ontology matching disabled.")
            self._built = True
            return self

        texts = [e.match_text() for e in self.entries]
        embs = np.asarray(self._model.encode(texts, normalize_embeddings=True))
        for entry, emb in zip(self.entries, embs):
            entry.embedding = emb
        self._embeddings = embs

        logger.info(
            "Ontology registry built: %d ontologies embedded (%s).",
            len(self.entries),
            ", ".join(f"{e.name}[{len(e.labels)}]" for e in self.entries[:8]),
        )
        self._built = True
        return self

    def is_ready(self) -> bool:
        return bool(self._built and self.entries and self._embeddings is not None)

    def match(self, text: str) -> Tuple[Optional[OntologyEntry], float]:
        """Return the best-matching ontology and its cosine similarity.

        Returns ``(None, best_similarity)`` when the best similarity is below
        ``match_threshold`` so the caller can fall back to a default label set.
        """
        if not self.is_ready() or not text or not text.strip():
            return None, 0.0

        query = np.asarray(self._model.encode([text], normalize_embeddings=True))[0]
        sims = self._embeddings @ query  # cosine (both normalised)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.match_threshold:
            return None, best_sim
        return self.entries[best_idx], best_sim
