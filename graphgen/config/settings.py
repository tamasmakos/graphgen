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
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from graphgen.config.schema import get_default_schema

class InfrastructureSettings(BaseSettings):
    """
    External Integration Settings.
    Crucial for connecting services. Managed via .env / docker-compose.
    """
    graph_db_type: str = Field("falkordb", alias="GRAPH_DB_TYPE")

    # --- Databases ---
    falkordb_host: str = Field("falkordb", alias="FALKORDB_HOST")
    falkordb_port: int = Field(6379, alias="FALKORDB_PORT")
    
    neo4j_host: str = Field("neo4j", alias="NEO4J_HOST")
    neo4j_port: int = Field(7687, alias="NEO4J_PORT")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("password", alias="NEO4J_PASSWORD")
    
    postgres_host: str = Field("pgvector", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("graphknows", alias="POSTGRES_DB")
    postgres_user: str = Field("postgres", alias="POSTGRES_USER")
    postgres_password: str = Field("password", alias="POSTGRES_PASSWORD")
    postgres_enabled: bool = True
    postgres_table: str = "hybrid_embeddings"
    
    # --- API Keys ---
    groq_api_key: Optional[SecretStr] = Field(None, alias="GROQ_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")
    
    # --- Filesystem (Docker Volumes) ---
    input_dir: str = Field("/app/input", alias="INPUT_DIR")
    output_dir: str = Field("/app/output", alias="OUTPUT_DIR")
    
    clean_start: bool = Field(True, alias="CLEAN_START")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class LLMSettings(BaseSettings):
    """
    Model Configuration.
    Defaults are set here but can be overridden via .env for experimentation.
    """
    base_model: str = Field("llama-3.1-8b-instant", alias="GROQ_MODEL")
    extraction_model: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", alias="EXTRACTION_MODEL")
    summarization_model: str = Field("llama-3.1-8b-instant", alias="SUMMARISATION_MODEL")
    
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
    ontology_dir: str = "/app/input/ontology/cdm-4.13.2"  # Directory with RDF files
    namespace_filter: Optional[str] = None  # Filter to specific namespace prefix
    merge_with_manual: bool = True  # Merge with gliner_labels or replace
    top_level_only: bool = True  # Only include classes with no named parents
    min_subclasses: int = 0  # Only include classes with at least this many subclasses
    include_local_names: bool = True  # Use URI local names as fallback

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class IterativeSettings(BaseSettings):
    """
    Iterative Experiment Configuration.
    """
    enabled: bool = Field(False, alias="ITERATIVE_ENABLED")
    batch_size: int = Field(100, alias="ITERATIVE_BATCH_SIZE")
    iterations: int = Field(5, alias="ITERATIVE_ITERATIONS")
    random_seed: int = Field(42, alias="ITERATIVE_RANDOM_SEED")

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
    gliner_model: str = "urchade/gliner_medium-v2.1"
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


class KGESettings(BaseSettings):
    """
    PyKeen Knowledge Graph Embedding Configuration.
    
    When enabled, trains a KGE model on entity-relation triples and uses
    the resulting embeddings to compute edge weights for Leiden community detection.
    """
    enabled: bool = False  # Toggle KGE training and weighted communities
    model: str = "DistMult"  # PyKeen model type
    embedding_dim: int = 64
    learning_rate: float = 0.01
    num_epochs: int = 50
    early_stopping_patience: int = 5

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class AnalysisSettings(BaseSettings):
    """
    Statistical Analysis Configuration.
    
    Controls the topic/community embedding separation tests that verify
    whether communities form distinct semantic clusters.
    """
    topic_separation_test: bool = True  # Enable/disable statistical tests
    output_file: str = "topic_separation_stats.json"  # Output filename
    hierarchy_levels: List[str] = ["COMMUNITY", "SUBCOMMUNITY"]  # Levels to analyze

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore"
    )


class TestModeSettings(BaseSettings):
    """
    Test Mode Configuration.
    
    When enabled, limits the number of documents processed for faster testing.
    Set max_documents to 0 to process all documents (no limit).
    """
    enabled: bool = False  # Toggle test mode
    max_documents: int = 0  # 0 = no limit, process all documents

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
    kge: KGESettings = Field(default_factory=KGESettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    community: CommunitySettings = Field(default_factory=CommunitySettings)
    test_mode: TestModeSettings = Field(default_factory=TestModeSettings)
    iterative: IterativeSettings = Field(default_factory=IterativeSettings)
    
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
        Factory method to load settings from YAML and Environment.
        YAML takes precedence over Defaults. Environment takes precedence over YAML (for secrets).
        """
        from graphgen.config.loader import load_yaml_config
        
        # 1. Load YAML
        yaml_config = load_yaml_config(config_path)
        
        # 2. Initialize with merged data (Env will still override fields if using BaseSettings standard behavior, 
        # but to ensure YAML overrides defaults we pass it as kwargs)
        # However, BaseSettings prefers Env > Init Kwargs > Defaults.
        # So passing YAML as kwargs is correct for YAML > Defaults.
        # But we also want Env > YAML.
        
        # Strategy:
        # Create instance with YAML data.
        # Pydantic BaseSettings automatic env loading will override if env vars exist.
        
        return cls(_env_file=env_file, **yaml_config)