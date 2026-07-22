from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "verify-container-postcondition":
        raise SystemExit(
            "The postcondition entry point permits only verified run postconditions"
        )
    verifier_path = Path(__file__).resolve()
    source_root = verifier_path.parents[3]
    integrity_path = verifier_path.with_name("ubuntu_source_integrity.py")
    spec = importlib.util.spec_from_file_location(
        "lumen_ubuntu_source_integrity",
        integrity_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the image-baked source verifier")
    integrity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integrity)
    integrity.load_verified_attestation(source_root)
    # Isolated Python has no repository or working-directory import path. Add
    # only the image-baked source root after its complete closure is verified.
    sys.path.insert(0, str(source_root))
    from tools.fine_tuning.unsloth.ubuntu_pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
