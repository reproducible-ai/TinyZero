#!/usr/bin/env python3
"""Package TinyZero's one-step actor checkpoint for harness verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open


ARTIFACT_PATH = "artifacts/tinyzero-canary/model.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_safetensors(path: Path) -> None:
    with safe_open(path, framework="numpy", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if not keys:
            raise RuntimeError("actor checkpoint contains no tensors")
        for key in keys:
            if checkpoint.get_tensor(key).size == 0:
                raise RuntimeError(f"actor tensor is empty: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/TinyZero/treqs-tinyzero-canary/actor/global_step_1"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tinyzero-canary")
    )
    args = parser.parse_args()

    source = args.checkpoint_dir / "model.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"missing actor checkpoint: {source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "model.safetensors"
    shutil.copy2(source, artifact)
    verify_safetensors(artifact)

    digest = sha256(artifact)
    size = artifact.stat().st_size
    manifest = {
        "schema": "reproai.artifact/v1",
        "format": "safetensors",
        "sha256": digest,
        "sizeBytes": size,
        "loadVerified": True,
        "files": [
            {
                "path": ARTIFACT_PATH,
                "sha256": digest,
                "sizeBytes": size,
            }
        ],
    }
    result = {
        "schema": "reproai.result/v1",
        "checkpoint": ARTIFACT_PATH,
        "artifactSha256": digest,
        "artifactSizeBytes": size,
        "loadVerified": True,
        "steps": 1,
        "optimizerSteps": 1,
    }
    (args.output_dir / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("E2E_ARTIFACT=" + json.dumps(manifest, sort_keys=True))
    print("E2E_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
