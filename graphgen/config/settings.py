"""
Configuration Management.

This module rationalizes the setup by separating:
1. Infrastructure & Integration (External connections, handled via .env)
2. Application Logic (Internal tuning, handled via defaults here)

Usage:
- Use .env for: Hostnames, Ports, API Keys, Model Selection.
- Edit this file for: Chunk sizes, Extraction rules, Thresholds.
"""

from typing import List, Optional, Dict, Any
from pydantic import Field, SecretStr, field_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from graphgen.config.schema import get_default_schema

class InfrastructureSettings(BaseSettings):
    """
    External Integration Settings.
    Crucial for connecting services. Managed via .env / docker-compose.
    """
    graph_db_type: str = Field("neo4j", alias="GRAPH_DB_TYPE")

    # --- Databases ---
    neo4j_host: str = Field("neo4j", alias="NEO4J_HOST")
    neo4j_port: int = Field(7687, alias="NEO4J_PORT")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("password", alias="NEO4J_PASSWORD")
    
    # --- API Keys ---
    # OpenRouter is the sole LLM gateway. Accept the correctly-spelled var and
    # the commonly-mistyped "OPENROUTE_API_KEY".
    openrouter_api_key: Optional[SecretStr] = Field(
        None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "OPENROUTE_API_KEY")
    )
    openai_api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")
    
    # --- Filesystem (Docker Volumes) ---
    input_dir: str = Field("/app/input", alias="INPUT_DIR")
    output_dir: str = Field("/app/output", alias="OUTPUT_DIR")
    
    clean_start: bool = Field(True, alias="CLEAN_START")
    neo4j_upload_enabled: bool = Field(True, alias="NEO4J_UPLOAD_ENABLED")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class LLMSettings(BaseSettings):
    """
    Model Configuration.
    Defaults are set here but can be overridden via .env for experimentation.
    """
    # OpenRouter model ids (provider/model). See https://openrouter.ai/models
    base_model: str = Field("meta-llama/llama-3.1-8b-instruct", alias="OPENROUTER_MODEL")
    extraction_model: str = Field("meta-llama/llama-3.3-70b-instruct", alias="EXTRACTION_MODEL")
    summarization_model: str = Field("meta-llama/llama-3.3-70b-instruct", alias="SUMMARIZATION_MODEL")
    
    temperature: float = 0.0
    max_retries: int = 3

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class OntologySettings(BaseSettings):
    """
    Ontology-based label extraction configuration.
    
    When enabled, extracts entity labels from RDF/OWL ontology files
    to use as GLiNER extraction labels. This enables domain-specific
    entity recognition based on ontology class definitions.
    """
    enabled: bool = False  # Toggle ontology label extraction
    ontology_dir: str = "input/ontology/cdm-4.13.2"  # Directory with RDF files
    namespace_filter: Optional[str] = None  # Filter to specific namespace prefix
    merge_with_manual: bool = True  # Merge with gliner_labels or replace
    top_level_only: bool = True  # Only include classes with no named parents
    min_subclasses: int = 0  # Only include classes with at least this many subclasses
    include_local_names: bool = True  # Use URI local names as fallback

    # Semantic per-chunk ontology matching (backend: "ontology_dspy")
    max_labels: int = 60  # Max class labels passed to the LLM per matched ontology
    match_threshold: float = 0.10  # Min cosine sim to accept an ontology match

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )



class ExtractionSettings(BaseSettings):
    """
    Internal Extraction Logic.
    Tuned for the specific data domain. Not typically in .env.
    """
    # Text Splitting
    chunk_size: int = 1200
    chunk_overlap: int = 100
    
    # Extraction Backend preference
    backend: str = "llm"  # options: "gliner", "spacy", "llm"
    
    # GLiNER Configuration
    gliner_model: str = "knowledgator/gliner-multitask-large-v0.5"
    gliner_threshold: float = 0.5
    device: str = "auto" # "auto", "cuda", "cpu"
    use_onnx: bool = False
    # Entity labels (used by GLiNER, Spacy hints, and LLM categories)
    entity_labels: List[str] = Field(default_factory=list, alias="gliner_labels")

    @field_validator("entity_labels", mode="before")
    @classmethod
    def validate_entity_labels(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item) for item in v]
        return []
    
    # Ontology-based label extraction
    ontology: OntologySettings = Field(default_factory=OntologySettings)
    
    # Spacy Configuration
    spacy_model: str = "en_core_web_lg"
    
    # Performance
    max_concurrent_chunks: int = 8
    
    # File Selection (for incremental/selective processing)
    file_pattern: str = Field("*.txt", alias="EXTRACTION_FILE_PATTERN")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class ProcessingSettings(BaseSettings):
    """
    Internal Graph Processing Logic.
    """
    # Graph Pruning
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    prune_isolated_nodes: bool = True
    min_component_size: int = 3
    
    # Similarity & Resolution
    similarity_threshold: float = 0.95

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class EmbeddingSettings(BaseSettings):
    """
    Embedding Model Configuration.
    """
    model_name: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")
    device: str = Field("auto", alias="EMBEDDING_DEVICE")
    cache_folder: Optional[str] = Field(None, alias="EMBEDDING_CACHE_FOLDER")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class AnalyticsSettings(BaseSettings):
    """
    Analytics & Visualization Configuration.

    Controls the generation of the thesis analytics report (modularity
    comparison, Node2Vec permutation test, similarity distribution) and
    optional visualizations.
    """
    enabled: bool = Field(False, alias="ANALYTICS_ENABLED")
    output_dir: str = "analytics_reports"

    # Visualization
    visualization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "interactive": True,
            "heatmap": True,
        }
    )

    # Provenance & artifact outputs
    outputs_subdir: str = "thesis_outputs"
    save_provenance: bool = True
    save_raw_overlap_matrix: bool = True

    # Optional path to a JSON file of ground-truth themes (list of
    # {id, name, description}) for quantitative topic validation.  When
    # None, the built-in EPRS theme set is used.
    ground_truth_themes_path: Optional[str] = None

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class TestModeSettings(BaseSettings):
    """
    Test Mode Configuration.

    When enabled, limits the total number of segments processed across all
    documents. Set max_segments to 0 to process all segments (no limit).
    """
    enabled: bool = False  # Toggle test mode
    max_segments: int = 0  # 0 = no limit, process all segments

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )



class CommunitySettings(BaseSettings):
    """
    Leiden Community Detection Configuration.
    """
    resolutions: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]
    n_iterations: int = 10  # Number of iterations for consistency
    min_community_size: int = 3  # Merge communities smaller than this
    seed: Optional[int] = 42
    
    # Node2Vec Weighting
    node2vec_enabled: bool = False
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 16
    node2vec_num_walks: int = 20

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class PipelineSettings(BaseSettings):
    """
    Master Configuration Object.
    Aggregates all specific settings groups.
    """
    infra: InfrastructureSettings = Field(default_factory=InfrastructureSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    community: CommunitySettings = Field(default_factory=CommunitySettings)
    test_mode: TestModeSettings = Field(default_factory=TestModeSettings)

    # Global/Runtime flags
    debug: bool = False
    
    # Schema is dynamic and not part of env settings usually
    # We can store it here or separately. Storing here for convenience.
    schema_config: Dict[str, Any] = Field(default_factory=lambda: get_default_schema().model_dump(), alias="schema") 

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    @classmethod
    def load(cls, config_path: str = "config.yaml", env_file: str = ".env") -> "PipelineSettings":
        """
        Factory method to load settings from YAML and environment variables.

        Priority (highest first): environment variables > .env file > config.yaml > field defaults.

        Each nested settings class is constructed via its own constructor so that
        BaseSettings env-var reading fires correctly for each sub-model. Passing nested
        dicts through PipelineSettings(**yaml_dict) coerces them via model_validate(),
        which bypasses the BaseSettings env-reading machinery for those nested classes.
        """
        from graphgen.config.loader import load_yaml_config

        yaml_config = load_yaml_config(config_path)

        nested: Dict[str, Any] = {
            "infra": InfrastructureSettings(**(yaml_config.pop("infra", None) or {})),
            "llm": LLMSettings(**(yaml_config.pop("llm", None) or {})),
            "extraction": ExtractionSettings(**(yaml_config.pop("extraction", None) or {})),
            "processing": ProcessingSettings(**(yaml_config.pop("processing", None) or {})),
            "embedding": EmbeddingSettings(**(yaml_config.pop("embedding", None) or {})),
            "analytics": AnalyticsSettings(**(yaml_config.pop("analytics", None) or {})),
            "community": CommunitySettings(**(yaml_config.pop("community", None) or {})),
            "test_mode": TestModeSettings(**(yaml_config.pop("test_mode", None) or {})),
        }

        # yaml_config now only contains top-level scalar keys (e.g. debug).
        return cls(_env_file=env_file, **nested, **yaml_config)
