"""
Compile the ReSkillio ingestion pipeline to a KFP-compatible YAML file.

    python3 scripts/compile_pipeline.py [--output pipeline.yaml]

The compiled YAML can be:
  - submitted directly via scripts/submit_pipeline.py
  - uploaded to Vertex AI Pipelines in the GCP Console
  - stored in version control as the versioned pipeline spec
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kfp import compiler
from reskillio.vertex.pipeline import ingestion_pipeline

DEFAULT_OUTPUT = Path(__file__).parent.parent / "pipeline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile ReSkillio ingestion pipeline")
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Output YAML path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling pipeline → {output_path}")
    compiler.Compiler().compile(
        pipeline_func=ingestion_pipeline,
        package_path=str(output_path),
    )
    print(f"Done. Pipeline spec written to: {output_path}")


if __name__ == "__main__":
    main()
