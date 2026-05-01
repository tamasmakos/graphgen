"""Run GraphGen with a supplied config file path.

Kept minimal so experiments are config-driven instead of hardcoded in ad hoc scripts.
"""
import asyncio
import os
import sys
from pathlib import Path

from graphgen.config.settings import PipelineSettings
from graphgen.main import resolve_env_file
from graphgen.orchestrator import KnowledgePipeline
from graphgen.pipeline.entity_relation.extractors import get_extractor
from graphgen.utils.logging import configure_logging


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_with_config.py <config-path>")

    config_path = str(Path(sys.argv[1]).resolve())
    os.chdir(Path(config_path).parent)

    env_file = resolve_env_file()
    configure_logging()
    settings = PipelineSettings.load(config_path=config_path, env_file=env_file)
    configure_logging(debug=settings.debug)
    extractor = get_extractor(settings.model_dump())
    asyncio.run(KnowledgePipeline(settings=settings, uploader=None, extractor=extractor).run())


if __name__ == "__main__":
    main()
