"""GraphGen pipeline entry point."""

import asyncio
import logging
import sys

from dotenv import load_dotenv

from graphgen.config.settings import PipelineSettings
from graphgen.orchestrator import KnowledgePipeline
from graphgen.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

async def run_pipeline() -> None:
    configure_logging()
    try:
        settings = PipelineSettings.load() # loads from config.yaml and .env
        configure_logging(debug=settings.debug)

        logger.info("Initializing GraphGen Pipeline...")
        
        logger.info(f"Configuration loaded. Input: {settings.infra.input_dir}, Output: {settings.infra.output_dir}")
        
        from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
        logger.info("Initializing Neo4j Backend...")
        uploader = Neo4jGraphUploader(
            host=settings.infra.neo4j_host,
            port=settings.infra.neo4j_port,
            username=settings.infra.neo4j_user,
            password=settings.infra.neo4j_password,
            database="neo4j"
        )
            
        # Initialize Extractor
        from graphgen.pipeline.entity_relation.extractors import get_extractor
        extractor = get_extractor(settings.model_dump())
        
        # Check for Iterative Mode
        if settings.iterative.enabled:
            from graphgen.pipeline.iterative_orchestrator import IterativeOrchestrator
            logger.info("Starting Iterative Pipeline Orchestrator...")
            pipeline = IterativeOrchestrator(settings, uploader, extractor)
        else:
            # Initialize the standard pipeline
            pipeline = KnowledgePipeline(settings, uploader, extractor)
        
        # Run
        await pipeline.run()
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

def main() -> None:
    """Entry point for the console script."""
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()