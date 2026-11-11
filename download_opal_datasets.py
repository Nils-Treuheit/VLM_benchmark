#!/usr/bin/env python3
"""Download all OPAL-supported datasets and prepare ImageFolder train/test trees.

The script uses the official source files and official splits where available.
Downloads resume from ``.part`` files. Oxford-IIIT Pet reuses the archive
already staged at ``opal_datasets/oxford_iiit_pet/oxford-pets-ibc__rdr``.
Stanford Cars and Stanford Dogs are fetched from Hugging Face into ``HF_HOME``.

Images are symlinked into OPAL's ImageFolder layout, so source files are stored
once. The output is under ``/mnt/HDD1/Project_Data/public_datasets/opal_datasets``.
Install SciPy to parse MATLAB split/label files and Hugging Face ``datasets``
to fetch Stanford Cars.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets")
DATASET_NAMES = ("cub", "cars", "pets", "dogs", "flowers")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SOURCES = {
    "cub": {
        "archive": (
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            "97eceeb196236b17998738112f37df78",
        ),
    },
    "pets": {
        "images.tar.gz": (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        "annotations.tar.gz": (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    },
    "flowers": {
        "102flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "imagelabels.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "setid.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
    },
}


def allowed_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname == "vision.stanford.edu"
    )


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str | None = None) -> None:
    if not allowed_url(url):
        raise ValueError(f"Unapproved download URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "VLM-benchmark-OPAL-datasets/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", response.getcode())
            append = offset > 0 and status == 206
            if offset and not append:
                print(f"Server did not resume {destination.name}; restarting from byte 0")
            mode = "ab" if append else "wb"
            digest = hashlib.md5() if expected_md5 else None
            if digest and append:
                with partial.open("rb") as old:
                    for chunk in iter(lambda: old.read(1024 * 1024), b""):
                        digest.update(chunk)
            written = offset if append else 0
            with partial.open(mode) as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if digest:
                        digest.update(chunk)
                    if written and written % (128 * 1024 * 1024) < len(chunk):
                        print(f"  {destination.name}: {written / (1024**2):.0f} MiB")
        if expected_md5 and digest and digest.hexdigest() != expected_md5:
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {destination.name}: expected {expected_md5}, "
                f"got {digest.hexdigest()}"
            )
        partial.replace(destination)
    except urllib.error.HTTPError as exc:
        # Preserve the partial payload to allow a later retry/resume.
        raise RuntimeError(f"Download failed for {url}: HTTP {exc.code}") from exc
    except Exception:
        # A network interruption leaves the partial file intact for resuming.
        raise


def safe_target(root: Path, member_name: str) -> Path:
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe archive path: {member_name}")
    target = (root / Path(*pure.parts)).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise ValueError(f"Archive path escapes destination: {member_name}")
    return target


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            for member in members:
                safe_target(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Refusing link/special file in {archive.name}: {member.name}")
            bundle.extractall(destination)
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                safe_target(destination, member.filename)
                mode = member.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError(f"Refusing symlink in {archive.name}: {member.filename}")
            bundle.extractall(destination)
        return
    raise ValueError(f"Unrecognized or incomplete archive: {archive}")


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def ensure_download(name: str, url: str, raw_root: Path, md5: str | None = None) -> Path:
    archive = raw_root / name
    if archive.exists() and md5 and md5_file(archive) != md5:
        archive.unlink()
    if not archive.exists():
        print(f"Downloading {name}: {url}")
        download(url, archive, md5)
    elif md5:
        print(f"Using verified archive: {archive}")
    return archive


def load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("SciPy is required for official dataset split files; install scipy first") from exc
    try:
        return loadmat(path, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def mat_strings(value: Any) -> list[str]:
    import numpy as np

    if isinstance(value, bytes):
        return [value.decode("utf-8", "replace").strip()]
    if isinstance(value, str):
        return [value.strip()]
    array = np.asarray(value)
    if array.dtype.kind in "US":
        flat = [str(item) for item in array.ravel()]
        if flat and all(len(item) == 1 for item in flat):
            return ["".join(flat).strip()]
        return [item.strip() for item in flat if item.strip()]
    if array.dtype.kind == "O":
        result: list[str] = []
        for item in array.ravel():
            result.extend(mat_strings(item))
        return result
    return []


def mat_ints(value: Any) -> list[int]:
    import numpy as np

    array = np.asarray(value)
    return [int(item) for item in array.ravel()]


def find_named_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def link_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        return
    destination.symlink_to(source.resolve())


def add_records(records: list[tuple[Path, str]], split_root: Path) -> int:
    for source, class_name in records:
        link_image(source, split_root / class_name / source.name)
    return len(records)


def prepare_cub(data_root: Path) -> Path:
    root = data_root / "opal_datasets" / "cub_200_2011"
    raw = root / "raw"
    archive = ensure_download("CUB_200_2011.tgz", SOURCES["cub"]["archive"][0], raw,
                              SOURCES["cub"]["archive"][1])
    images_dir = raw / "CUB_200_2011" / "images"
    if count_images(images_dir) == 0:
        extract(archive, raw)
    base = raw / "CUB_200_2011"
    classes = {int(line.split()[0]): line.split(maxsplit=1)[1].replace("_", " ")
               for line in (base / "classes.txt").read_text().splitlines()}
    image_paths = {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1]
                   for line in (base / "images.txt").read_text().splitlines()}
    labels = {int(fields[0]): int(fields[1]) for fields in
              (line.split() for line in (base / "image_class_labels.txt").read_text().splitlines())}
    train_flags = {int(fields[0]): int(fields[1]) for fields in
                   (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
    out = root / "dataset"
    counts = {"train": [], "test": []}
    for image_id, relative in image_paths.items():
        split = "train" if train_flags[image_id] == 1 else "test"
        class_name = classes[labels[image_id]]
        counts[split].append((base / "images" / relative, class_name))
    for split, rows in counts.items():
        add_records(rows, out / split)
        print(f"CUB-200-2011 {split}: {len(rows)} images")
    return out


def _mat_field(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    if hasattr(record, key):
        return getattr(record, key)
    try:
        return record[key]
    except Exception as exc:
        raise KeyError(f"MAT record has no {key!r} field") from exc


def _mat_records(value: Any) -> list[Any]:
    import numpy as np

    if isinstance(value, (list, tuple)):
        return list(value)
    array = np.asarray(value, dtype=object)
    return list(array.ravel())


def prepare_cars(data_root: Path) -> Path:
    root = data_root / "opal_datasets" / "stanford_cars"
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required for Stanford Cars; "
            "install it in the active Python environment"
        ) from exc

    # The original Stanford file host is unreliable. This HF dataset mirrors
    # the official 8,144/8,041 split and uses HF_HOME for its download cache.
    source = load_dataset("mteb/StanfordCars")
    label_feature = source["train"].features["label"]
    out = root / "dataset"
    counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        rows = source[split].cast_column("image", HFImage(decode=False))
        for index, row in enumerate(rows):
            label = int(row["label"])
            # ImageFolder treats path separators in labels as nested classes.
            class_name = label_feature.int2str(label).replace("/", "-").replace("\\", "-")
            class_name = " ".join(class_name.split())
            image = row["image"]
            destination = out / split / class_name / f"{index:05d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = image.get("bytes")
            if payload is not None:
                destination.write_bytes(payload)
            else:
                source_path = image.get("path")
                if not source_path:
                    raise ValueError(f"Stanford Cars row {split}/{index} has no image payload")
                shutil.copyfile(source_path, destination)
            counts[split] += 1
        print(f"Stanford Cars {split}: {counts[split]} images")

    if counts != {"train": 8144, "test": 8041}:
        raise ValueError(f"Unexpected Stanford Cars split sizes: {counts}")
    return out


def prepare_pets(data_root: Path) -> Path:
    root = data_root / "opal_datasets" / "oxford_iiit_pet"
    # This path is retained from the previous partial download so its 120 MB
    # images.tar.gz.part can resume in place.
    raw = root / "oxford-pets-ibc__rdr"
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, md5) in SOURCES["pets"].items():
        archive = ensure_download(name, url, raw, md5)
        required = raw / ("images" if name.startswith("images") else "annotations")
        if not required.exists():
            extract(archive, raw)
    empty_old_layout = raw / "oxford-iiit-pet"
    if empty_old_layout.is_dir() and not any(empty_old_layout.iterdir()):
        empty_old_layout.rmdir()
    image_root = raw / "images"
    annotation_root = raw / "annotations"
    out = root / "dataset"
    counts = {"train": 0, "test": 0}
    for split, annotation in (("train", "trainval.txt"), ("test", "test.txt")):
        rows = []
        for line in (annotation_root / annotation).read_text().splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            image_id = fields[0]
            rows.append((image_root / f"{image_id}.jpg", image_id.rsplit("_", 1)[0]))
        counts[split] = add_records(rows, out / split)
        print(f"Oxford-IIIT Pets {split}: {counts[split]} images")
    return out


def prepare_dogs(data_root: Path) -> Path:
    root = data_root / "opal_datasets" / "stanford_dogs"
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required for Stanford Dogs; "
            "install it in the active Python environment"
        ) from exc

    # The Stanford host for the original tar files timed out. This HF mirror
    # states that its images and labels are unmodified Stanford Dogs data.
    source = load_dataset("RMASD/stanford-dogs")
    label_feature = source["train"].features["label"]
    out = root / "dataset"
    counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        rows = source[split].cast_column("image", HFImage(decode=False))
        for index, row in enumerate(rows):
            label = int(row["label"])
            class_name = row.get("class_name") or label_feature.int2str(label)
            image = row["image"]
            destination = out / split / class_name / f"{index:05d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = image.get("bytes")
            if payload is not None:
                destination.write_bytes(payload)
            else:
                source_path = image.get("path")
                if not source_path:
                    raise ValueError(f"Stanford Dogs row {split}/{index} has no image payload")
                shutil.copyfile(source_path, destination)
            counts[split] += 1
        print(f"Stanford Dogs {split}: {counts[split]} images")

    if counts != {"train": 12000, "test": 8580}:
        raise ValueError(f"Unexpected Stanford Dogs split sizes: {counts}")
    return out


def prepare_flowers(data_root: Path) -> Path:
    root = data_root / "opal_datasets" / "oxford_flowers_102"
    raw = root / "raw"
    for name, url in SOURCES["flowers"].items():
        archive = ensure_download(name, url, raw)
        expected = {"102flowers.tgz": "jpg", "imagelabels.mat": name,
                    "setid.mat": name}[name]
        if name.endswith(".tgz"):
            ready = list(raw.rglob(expected))
        else:
            ready = list(raw.rglob(name))
        if not ready:
            if name.endswith(".tgz"):
                extract(archive, raw)
            else:
                shutil.copyfile(archive, raw / name)
    image_root = next(path for path in raw.rglob("jpg") if path.is_dir())
    labels_mat = load_mat(find_named_file(raw, "imagelabels.mat"))
    splits_mat = load_mat(find_named_file(raw, "setid.mat"))
    labels = mat_ints(labels_mat["labels"])
    train_ids = mat_ints(splits_mat["trnid"]) + mat_ints(splits_mat["valid"])
    test_ids = mat_ints(splits_mat["tstid"])
    out = root / "dataset"
    for split, ids in (("train", train_ids), ("test", test_ids)):
        rows = []
        for image_id in ids:
            label = labels[image_id - 1]
            source = image_root / f"image_{image_id:05d}.jpg"
            rows.append((source, f"class_{label:03d}"))
        add_records(rows, out / split)
        print(f"Oxford Flowers 102 {split}: {len(rows)} images")
    return out


PREPARE = {
    "cub": prepare_cub,
    "cars": prepare_cars,
    "pets": prepare_pets,
    "dogs": prepare_dogs,
    "flowers": prepare_flowers,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("PROJECT_PUBLIC_DATASETS", DEFAULT_ROOT)),
        help="public_datasets root (default: %(default)s)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(*DATASET_NAMES, "all"),
        default=["all"],
        help="datasets to fetch and prepare (default: all five)",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    selected = DATASET_NAMES if "all" in args.datasets else tuple(dict.fromkeys(args.datasets))
    print(f"Destination root: {data_root / 'opal_datasets'}")
    for name in selected:
        print(f"\n=== {name} ===")
        target = PREPARE[name](data_root)
        print(f"ImageFolder ready: {target}")
    print("\nOPAL datasets are prepared. Train OPAL separately using its README and each dataset path above.")
    print("OD-LVIS and Roboflow Vision Evals are not downloaded by this script; neither has a verified public data archive URL.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
