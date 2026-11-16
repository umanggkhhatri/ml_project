"""Download a pinned official release and extract without path traversal."""

import hashlib
import zipfile
from pathlib import Path

import requests

from shapeguard.reproducibility import save_json, sha256


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is not allowed: {info.filename}")
        zf.extractall(destination)


def download_fracatlas(root="data/raw", version=7):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / f"fracatlas-v{version}.json"
    response = requests.get(
        f"https://api.figshare.com/v2/articles/22363012/versions/{version}", timeout=60
    )
    response.raise_for_status()
    metadata = response.json()
    file = next(f for f in metadata["files"] if f["name"].lower().endswith(".zip"))
    archive = root / f"fracatlas-v{version}.zip"
    expected_md5 = file.get("computed_md5")

    def verified(path):
        if not path.exists() or path.stat().st_size != file["size"]:
            return False
        if not expected_md5:
            return True
        digest = hashlib.md5()  # Source integrity checksum, not a security primitive.
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == expected_md5

    if not verified(archive):
        temporary = archive.with_suffix(".partial")
        with requests.get(file["download_url"], stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            with temporary.open("wb") as handle:
                for block in r.iter_content(1024 * 1024):
                    handle.write(block)
        if not verified(temporary):
            raise ValueError("Downloaded archive failed source size/checksum verification")
        temporary.replace(archive)
    destination = root / f"fracatlas-v{version}"
    marker = destination / ".extracted.json"
    checksum = sha256(archive)
    if not marker.exists():
        extract_zip(archive, destination)
        save_json(marker, {"archive_sha256": checksum})
    save_json(metadata_path, {"metadata": metadata, "archive_sha256": checksum})
    return destination
