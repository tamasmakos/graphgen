import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

from graphgen.config.settings import PipelineSettings
from graphgen.utils.graphdb.uploader import KnowledgeGraphUploader
from graphgen.orchestrator import KnowledgePipeline
from dotenv import load_dotenv

# Load env variables
load_dotenv()

async def main():
    try:
        logger.info("Initializing GraphGen Pipeline...")
        settings = PipelineSettings.load() # loads from config.yaml and .env
        
        logger.info(f"Configuration loaded. Input: {settings.infra.input_dir}, Output: {settings.infra.output_dir}")
        
        uploader = None
        if settings.infra.graph_db_type == "neo4j":
            from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
            logger.info("Initializing Neo4j Backend...")
            uploader = Neo4jGraphUploader(
                host=settings.infra.neo4j_host,
                port=settings.infra.neo4j_port,
                username=settings.infra.neo4j_user,
                password=settings.infra.neo4j_password,
                database="neo4j"
            )
        else:
            logger.info("Initializing FalkorDB Backend...")
            postgres_config = None
            if settings.infra.postgres_enabled:
                postgres_config = {
                    'enabled': True,
                    'host': settings.infra.postgres_host,
                    'port': settings.infra.postgres_port,
                    'user': settings.infra.postgres_user,
                    'password': settings.infra.postgres_password,
                    'database': settings.infra.postgres_db,
                    'table_name': settings.infra.postgres_table
                }
            
            uploader = KnowledgeGraphUploader(
                host=settings.infra.falkordb_host,
                port=settings.infra.falkordb_port,
                database="kg",
                postgres_config=postgres_config
            )
            
        # Initialize Extractor
        from graphgen.pipeline.entity_relation.extractors import get_extractor
        extractor = get_extractor(settings.model_dump())
        
        # Initialize the pipeline
        pipeline = KnowledgePipeline(settings, uploader, extractor)
        
        # Run
        await pipeline.run()
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
