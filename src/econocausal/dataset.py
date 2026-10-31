"""Download and extract a dataset described by repository metadata."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen
import zipfile


def load_metadata(metadata_path: str | Path) -> dict[str, object]:
    """Load and minimally validate a dataset metadata JSON file."""
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "download_url", "source_page", "license"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Dataset metadata missing keys: {missing}")
    if not isinstance(metadata["license"], dict) or not metadata["license"].get("url"):
        raise ValueError("Dataset metadata must include license.url")
    return metadata


def download_dataset(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Download the metadata-described ZIP and extract it into ``output_dir``.

    The archive is kept alongside extracted files so the run is auditable. Existing
    files are overwritten only for the archive itself; extraction rejects unsafe
    paths that could escape the output directory.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    metadata = load_metadata(metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "dataset.zip"
    with urlopen(str(metadata["download_url"]), timeout=timeout) as response:
        archive_path.write_bytes(response.read())

    root = output.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        archive.extractall(output)
        extracted = sorted(member.filename for member in archive.infolist() if not member.is_dir())

    return {
        "dataset": metadata["name"],
        "archive": str(archive_path),
        "output_dir": str(output),
        "extracted_files": extracted,
        "source_page": metadata["source_page"],
        "license": metadata["license"],
    }
