import dspy
from typing import Any, List
from pydantic import BaseModel, Field, field_validator

# Delimiter used when an LLM returns a JSON list for a string entity field.
# e.g. target: ["RUSSIA", "CHINA"] → "RUSSIA|||CHINA"
# The extractor's _parse_dspy_triplets expands these back into individual triplets.
ENTITY_LIST_SEPARATOR = "|||"


class Triplet(BaseModel):
    source: str = Field(description="The source entity")
    source_type: str = Field(description="The ontology class of the source entity")
    relation: str = Field(description="The relationship between source and target")
    target: str = Field(description="The target entity")
    target_type: str = Field(description="The ontology class of the target entity")
    confidence: float = Field(default=1.0, description="Confidence score (0.0-1.0) of this extraction")
    evidence: str = Field(default="", description="Verbatim text snippet that supports this relation")

    @field_validator("source", "target", mode="before")
    @classmethod
    def coerce_entity_list(cls, v: Any) -> str:
        """
        LLMs sometimes emit a JSON array for entity fields when a single relation
        maps to multiple targets (fan-out).  Rather than crashing Pydantic validation,
        join all values with ENTITY_LIST_SEPARATOR so the information is preserved.
        _parse_dspy_triplets in extractors.py then splits this back into individual
        triplets via a cartesian product expansion.
        """
        if isinstance(v, list):
            return ENTITY_LIST_SEPARATOR.join(str(item).strip() for item in v if item)
        return v

class EntityRelation(dspy.Signature):
    """
    Extract knowledge graph triplets (source, relation, target) from text.
    Use the provided ontology classes to filter allowed entity types if applicable, but primarily focus on the text.
    Use the provided entity hints as a guide for what entities might be present.
    ONLY extract relations that are explicitly stated in the text and have high confidence.
    Avoid "fluff" or trivial relations. Focus on significant interactions, facts, and properties.
    """
    text: str = dspy.InputField(desc="The input text to extract relations from.")
    ontology_classes: List[str] = dspy.InputField(desc="List of allowed ontology classes/types for entities.")
    entity_hints: List[str] = dspy.InputField(desc="List of pre-identified entities to guide extraction.")
    triplets: List[Triplet] = dspy.OutputField(desc="List of extracted triplets.")

class GraphExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityRelation)

    def forward(self, text: str, ontology_classes: List[str], entity_hints: List[str]):
        return self.extract(text=text, ontology_classes=ontology_classes, entity_hints=entity_hints)


# ---------------------------------------------------------------------------
# Ontology-guided extraction
# ---------------------------------------------------------------------------


class KnowledgeTriplet(BaseModel):
    """A single knowledge-graph fact extracted from text: (source)-[relation]->(target).

    Every field is grounded in the input chunk.  The ``source_type`` and
    ``target_type`` MUST be drawn from the ontology label set supplied to the
    extractor — this is what keeps the graph consistent with the matched
    ontology.
    """

    source: str = Field(
        description="The source entity of the relation, as a concise noun phrase "
        "(e.g. 'European Central Bank'). Use the canonical surface form from the text."
    )
    source_type: str = Field(
        description="The ontology class of the source entity. MUST be one of the "
        "provided ontology labels."
    )
    relation: str = Field(
        description="A short, active-voice label for how source relates to target "
        "(e.g. 'imposes', 'is member of', 'funds'). Derive it from the text, not a fixed list."
    )
    target: str = Field(
        description="The target entity of the relation, as a concise noun phrase."
    )
    target_type: str = Field(
        description="The ontology class of the target entity. MUST be one of the "
        "provided ontology labels."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in [0,1] that this relation is explicitly stated in the text.",
    )
    evidence: str = Field(
        default="",
        description="A short verbatim snippet from the chunk that supports this relation.",
    )

    @field_validator("source", "target", mode="before")
    @classmethod
    def coerce_entity_list(cls, v: Any) -> str:
        """Preserve fan-out when the LLM emits a list for an entity field.

        See :class:`Triplet.coerce_entity_list`; the shared delimiter lets
        ``_parse_dspy_triplets`` expand multi-valued entities into individual
        triplets via a cartesian product.
        """
        if isinstance(v, list):
            return ENTITY_LIST_SEPARATOR.join(str(item).strip() for item in v if item)
        return v


class OntologyGuidedExtraction(dspy.Signature):
    """Extract knowledge-graph triplets from a text chunk, constrained to one ontology.

    The chunk has already been semantically matched to the single most
    relevant domain ontology.  Use the ontology description to understand the
    domain framing, and restrict every entity's type to the provided ontology
    labels.  Extract ONLY relations explicitly stated in the text, with the
    entity types assigned from the allowed labels. Do not invent types outside
    the provided label set, and skip trivial or purely rhetorical statements.
    """

    text: str = dspy.InputField(desc="The text chunk to extract knowledge triplets from.")
    ontology_description: str = dspy.InputField(
        desc="A natural-language description of the ontology this chunk was matched to, "
        "establishing the domain and framing for extraction."
    )
    ontology_labels: List[str] = dspy.InputField(
        desc="The closed set of allowed entity types (ontology classes). Every "
        "source_type and target_type must be chosen from this list."
    )
    triplets: List[KnowledgeTriplet] = dspy.OutputField(
        desc="The list of knowledge triplets extracted from the text, typed against "
        "the ontology labels."
    )


class OntologyGuidedExtractorModule(dspy.Module):
    """dspy.Predict wrapper around :class:`OntologyGuidedExtraction`."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(OntologyGuidedExtraction)

    def forward(self, text: str, ontology_description: str, ontology_labels: List[str]):
        return self.extract(
            text=text,
            ontology_description=ontology_description,
            ontology_labels=ontology_labels,
        )
