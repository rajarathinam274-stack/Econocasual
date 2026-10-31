import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from econocausal.dataset import download_dataset, load_metadata


def write_metadata(path: Path, download_url: str) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "Example",
                "download_url": download_url,
                "source_page": "https://example.test/source",
                "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            }
        ),
        encoding="utf-8",
    )


def test_download_dataset_extracts_archive_and_returns_provenance(tmp_path):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("online_retail.csv", "Quantity,UnitPrice\n2,5\n")
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata, archive.as_uri())

    result = download_dataset(metadata, tmp_path / "raw")

    assert (tmp_path / "raw" / "dataset.zip").is_file()
    assert (tmp_path / "raw" / "online_retail.csv").read_text() == "Quantity,UnitPrice\n2,5\n"
    assert result["extracted_files"] == ["online_retail.csv"]
    assert result["license"]["name"] == "CC BY 4.0"


def test_download_dataset_rejects_archive_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside.txt", "must not extract")
    metadata = tmp_path / "metadata.json"
    write_metadata(metadata, archive.as_uri())

    with pytest.raises(ValueError, match="Unsafe archive member path"):
        download_dataset(metadata, tmp_path / "raw")
    assert not (tmp_path / "outside.txt").exists()


def test_load_metadata_requires_license_url(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "name": "Example",
                "download_url": "https://example.test/data.zip",
                "source_page": "https://example.test/source",
                "license": {"name": "Unknown"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="license.url"):
        load_metadata(metadata)
