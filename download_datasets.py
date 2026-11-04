"""Download optional datasets used by the VLM benchmark.

The LVIS archives are validated before they replace an existing file. This is
important because the upstream host can return an HTML error page with HTTP
200, which must not be mistaken for a successful ZIP download.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(os.getenv("VLM_DATASETS_DIR", "/mnt/HDD1/Project_Data/public_datasets"))
LVIS_DIR = BASE_DIR / "lvis"
LVIS_URLS = {
    "train": (
        "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    ),
    "val": (
        "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
        "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    ),
}
HF_DEFAULT_REPO = "redactable-llm/TextVQA"
HF_DEFAULT_FILE = "TextVQA_0.5.1_val.json"


def _valid_lvis_zip(path: Path, split: str) -> bool:
    """Return true only for a complete archive with the expected LVIS identity."""
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    expected = f"lvis_v1_{split}.json"
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if Path(name).name == expected]
            if len(members) != 1 or archive.testzip() is not None:
                return False

            # Read only the small metadata object, not the multi-GB expanded JSON.
            with archive.open(members[0]) as annotation:
                prefix = annotation.read(64 * 1024).decode("utf-8")
            match = re.search(r'"info"\s*:\s*', prefix)
            if not match:
                return False
            metadata, _ = json.JSONDecoder().raw_decode(prefix, match.end())
            if not isinstance(metadata, dict):
                return False
            description = metadata.get("description", "")
            version = str(metadata.get("version", ""))
            return isinstance(description, str) and "LVIS" in description and version in {"1", "1.0"}
    except (OSError, zipfile.BadZipFile):
        return False
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False


def _download_lvis_archive(split: str, destination: Path) -> bool:
    """Download and atomically install one validated LVIS annotation archive."""
    if _valid_lvis_zip(destination, split):
        print(f"[=] Valid LVIS archive already present: {destination}")
        return True
    if destination.exists():
        print(f"[!] Existing file is not a valid LVIS ZIP; replacing after a valid download: {destination}")

    try:
        import requests
    except ImportError:
        print("[X] requests is required for LVIS downloads (install it with `pip install requests`).")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in LVIS_URLS[split]:
        temp_name: str | None = None
        try:
            print(f"[+] Downloading {url}")
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                expected_size_header = response.headers.get("content-length")
                expected_size = int(expected_size_header) if expected_size_header else None
                etag = response.headers.get("etag", "").strip('"')
                md5_etag = etag if re.fullmatch(r"[0-9a-fA-F]{32}", etag) else None
                received_size = 0
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
                ) as temp_file:
                    temp_name = temp_file.name
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            temp_file.write(chunk)
                            received_size += len(chunk)

            temp_path = Path(temp_name)
            if expected_size is not None and received_size != expected_size:
                raise ValueError(f"incomplete response: expected {expected_size} bytes, received {received_size}")
            if md5_etag:
                import hashlib

                digest = hashlib.md5()
                with temp_path.open("rb") as downloaded:
                    for chunk in iter(lambda: downloaded.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest().lower() != md5_etag.lower():
                    raise ValueError("download identity does not match the server MD5 ETag")
            if not _valid_lvis_zip(temp_path, split):
                raise ValueError("response was incomplete or did not contain the expected LVIS split")
            os.replace(temp_path, destination)
            print(f"[✓] Installed validated LVIS {split} archive: {destination}")
            return True
        except Exception as exc:  # Report each mirror failure, then try the next.
            errors.append(f"{urlparse(url).netloc}: {exc}")
            print(f"[X] Download failed: {errors[-1]}")
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)

    print(f"[X] Could not download LVIS {split}: {'; '.join(errors)}")
    return False


def download_lvis() -> bool:
    """Fetch both LVIS annotation archives, preserving existing valid files."""
    outcomes = [
        _download_lvis_archive(split, LVIS_DIR / f"lvis_v1_{split}.json.zip")
        for split in ("train", "val")
    ]
    return all(outcomes)


def _valid_rf100_dataset(path: Path) -> bool:
    """Check that an extracted RF100-VL project has usable COCO data."""
    if not path.is_dir():
        return False

    found_annotations = False
    for annotation_path in path.glob("*/_annotations.coco.json"):
        try:
            with annotation_path.open(encoding="utf-8") as handle:
                annotations = json.load(handle)
            images = annotations.get("images")
            categories = annotations.get("categories")
            objects = annotations.get("annotations")
            if not isinstance(images, list) or not isinstance(categories, list) or not isinstance(objects, list):
                return False
            if not images or not categories:
                return False
            if any(
                not isinstance(image, dict)
                or not isinstance(image.get("file_name"), str)
                or not (annotation_path.parent / image["file_name"]).is_file()
                for image in images
            ):
                return False
            found_annotations = True
        except (OSError, json.JSONDecodeError):
            return False
    return found_annotations


def download_roboflow_group() -> bool:
    """Keep complete RF100-VL projects and download only missing or broken ones."""
    target = BASE_DIR / "rf100_vl"
    target.mkdir(parents=True, exist_ok=True)

    existing_projects = [path for path in target.iterdir() if path.is_dir()]
    if len(existing_projects) == 100 and all(_valid_rf100_dataset(path) for path in existing_projects):
        print(f"[=] All 100 RF100-VL datasets are already complete: {target}")
        return True

    if not os.getenv("ROBOFLOW_API_KEY"):
        print("[X] RF100-VL has missing or incomplete data, but ROBOFLOW_API_KEY is not set.")
        return False
    try:
        from rf100vl import get_rf100vl_projects
    except ImportError:
        print("[X] RF100-VL requires the official `rf100vl` package.")
        return False

    try:
        projects = get_rf100vl_projects()
        incomplete = [dataset for dataset in projects if not _valid_rf100_dataset(target / dataset.name)]
        for dataset in incomplete:
            destination = target / dataset.name
            print(f"[+] Downloading missing or incomplete RF100-VL dataset: {dataset.name}")
            dataset.download(str(destination), model_format="coco", overwrite=True)
            if not _valid_rf100_dataset(destination):
                raise ValueError(f"download did not produce a complete COCO dataset: {destination}")
        print(f"[✓] RF100-VL datasets are complete: {target}")
        return True
    except Exception as exc:
        print(f"[X] RF100-VL download failed: {exc}")
        return False


def download_huggingface_group(repo_id: str | None, filename: str | None) -> bool:
    """Ensure the TextVQA validation annotations are present and valid."""
    token = os.getenv("HF_TOKEN")
    if not repo_id or not filename:
        print("[X] Set both --hf-repo and --hf-file; refusing an incomplete Hugging Face asset selection.")
        return False
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError:
        print("[X] Hugging Face downloads require `huggingface_hub`.")
        return False

    target = BASE_DIR / "hf_datasets"
    local_file = target / filename
    is_default_asset = repo_id == HF_DEFAULT_REPO and filename == HF_DEFAULT_FILE
    cached_file = try_to_load_from_cache(repo_id, filename, repo_type="dataset") if is_default_asset else None
    if is_default_asset and isinstance(cached_file, str) and _valid_textvqa_annotations(Path(cached_file)):
        if _valid_textvqa_annotations(local_file):
            print(f"[=] Valid TextVQA annotations are present locally and in HF_HOME cache: {local_file}")
            return True
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target, prefix=f".{filename}.", suffix=".part", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            shutil.copyfile(cached_file, temp_path)
            os.replace(temp_path, local_file)
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"[=] Reused validated Hugging Face cache entry: {cached_file}")
        return True
    if is_default_asset:
        print("[=] No valid TextVQA annotation file in the HF_HOME cache; checking the dataset root.")
    if is_default_asset and _valid_textvqa_annotations(local_file):
        print(f"[=] Valid Hugging Face TextVQA annotations already present: {local_file}")
        return True

    target.mkdir(parents=True, exist_ok=True)
    try:
        result = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(target),
        )
        if repo_id == HF_DEFAULT_REPO and filename == HF_DEFAULT_FILE and not _valid_textvqa_annotations(Path(result)):
            raise ValueError("downloaded file is not a valid TextVQA validation annotation file")
        print(f"[✓] Downloaded {repo_id}/{filename}: {result}")
        return True
    except Exception as exc:
        print(f"[X] Hugging Face download failed for {repo_id}/{filename}: {exc}")
        return False


def _valid_textvqa_annotations(path: Path) -> bool:
    """Reject missing files and HTML error pages masquerading as annotations."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != 5000:
            return False
        if payload.get("dataset_name", "textvqa").casefold() != "textvqa":
            return False
        if payload.get("dataset_type", "val").casefold() not in {"val", "validation"}:
            return False
        return all(
            isinstance(row, dict)
            and isinstance(row.get("question"), str)
            and isinstance(row.get("image_id"), str)
            and isinstance(row.get("answers"), list)
            for row in rows
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lvis", action="store_true", help="Do not download LVIS annotations.")
    parser.add_argument("--hf-repo-skip", action="store_true", help="Skip the Hugging Face dataset group.")
    parser.add_argument("--hf-file-skip", action="store_true", help="Skip the Hugging Face dataset file group.")
    parser.add_argument("--rf100-skip", action="store_true", help="Skip RF100-VL.")
    parser.add_argument("--hf-repo", default=HF_DEFAULT_REPO, help="Hugging Face dataset repository ID.")
    parser.add_argument("--hf-file", default=HF_DEFAULT_FILE, help="File path within the selected Hugging Face dataset repository.")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using dataset root: {BASE_DIR}")
    print("COCO is expected to be present locally; this script does not download COCO.")

    outcomes = []
    if not args.skip_lvis:
        outcomes.append(download_lvis())
    if not (args.hf_repo_skip and args.hf_file_skip):
        outcomes.append(download_huggingface_group(args.hf_repo, args.hf_file))
    if not args.rf100_skip:
        outcomes.append(download_roboflow_group())

    if outcomes and not all(outcomes):
        print("One or more selected downloads failed; see errors above.")
        return 1
    print("All enabled dataset groups are present and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
