#!/usr/bin/env python3
"""Create deterministic, class-covering OPAL test panels for VLM comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {
    "cub_200_2011": {"display_name": "CUB-200-2011", "classes": 200, "test": 5794},
    "stanford_cars": {"display_name": "Stanford Cars", "classes": 196, "test": 8041},
    "oxford_iiit_pet": {"display_name": "Oxford-IIIT Pets", "classes": 37, "test": 3669},
    "stanford_dogs": {"display_name": "Stanford Dogs", "classes": 120, "test": 8580},
    "oxford_flowers_102": {"display_name": "Oxford Flowers 102", "classes": 102, "test": 6149},
}
DEFAULT_DATA_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/opal_datasets")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "opal_test_panels.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Category order is the official 1-based Oxford Flowers label order used by
# TensorFlow Datasets; source: tensorflow_datasets/datasets/oxford_flowers102.
FLOWER_NAMES = (
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
)


def stable_key(seed: int, rel_path: str) -> str:
    return hashlib.sha256(f"{seed}:{rel_path}".encode("utf-8")).hexdigest()


def prompt_label(dataset_name: str, class_name: str) -> str:
    if dataset_name == "cub_200_2011":
        # The leading three-digit value is a folder index, not part of the
        # bird's common name. Keep it in class_name for auditability, but omit
        # it from the zero-shot classification prompt.
        return class_name.split(".", 1)[-1].replace("_", " ").strip()
    if dataset_name == "stanford_dogs":
        return class_name.split("-", 1)[-1].replace("_", " ")
    if dataset_name == "oxford_iiit_pet":
        return class_name.replace("_", " ")
    if dataset_name == "oxford_flowers_102":
        index = int(class_name.removeprefix("class_"))
        return FLOWER_NAMES[index - 1]
    return class_name


def choose_panel(dataset_root: Path, dataset_name: str, info: dict[str, object],
                 sample_count: int, seed: int) -> dict[str, object]:
    test_root = dataset_root / "dataset" / "test"
    grouped: dict[str, list[Path]] = {}
    for path in test_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            class_name = path.parent.relative_to(test_root).as_posix()
            grouped.setdefault(class_name, []).append(path)
    if len(grouped) != info["classes"]:
        raise ValueError(f"{dataset_name}: expected {info['classes']} classes, found {len(grouped)}")

    per_class: list[tuple[str, list[Path]]] = []
    for class_name in sorted(grouped):
        images = sorted(
            grouped[class_name],
            key=lambda p: stable_key(seed, p.relative_to(dataset_root).as_posix()),
        )
        if not images:
            raise ValueError(f"{dataset_name}: no test images for class {class_name}")
        per_class.append((class_name, images))

    available_count = sum(len(images) for _, images in per_class)
    if available_count != info["test"]:
        raise ValueError(f"{dataset_name}: expected {info['test']} test images, found {available_count}")
    if sample_count < len(per_class):
        raise ValueError(f"{dataset_name}: sample_count {sample_count} would omit test classes")
    if sample_count > available_count:
        raise ValueError(f"{dataset_name}: sample_count {sample_count} exceeds test set size")

    chosen: list[tuple[Path, Path]] = []
    # Give every class one image first. Add one image per class per round so
    # class counts differ by at most one, including datasets with <200 classes.
    offsets = [0] * len(per_class)
    while len(chosen) < sample_count:
        added = False
        for index, (class_name, images) in enumerate(per_class):
            if offsets[index] >= len(images):
                continue
            chosen.append((class_name, images[offsets[index]]))
            offsets[index] += 1
            added = True
            if len(chosen) == sample_count:
                break
        if not added:
            raise RuntimeError(f"{dataset_name}: unable to fill the requested panel")

    classes = [class_name for class_name, _ in per_class]
    label_indices = {name: index for index, name in enumerate(classes)}
    examples = []
    for class_name, image in chosen:
        image_path = image.resolve()
        examples.append({
            "id": f"{dataset_name}:{class_name}:{image.name}",
            "image": str(image_path),
            "class_name": class_name,
            "prompt_label": prompt_label(dataset_name, class_name),
            "label": label_indices[class_name],
        })

    return {
        "dataset": info["display_name"],
        "dataset_dir": str(dataset_root.resolve()),
        "split": "official test",
        "sample_count": sample_count,
        "class_count": len(classes),
        "seed": seed,
        "classes": classes,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "protocol": "fixed class-covering 200-image panel; round-robin equalized across sorted classes",
        "sample_count_per_dataset": args.sample_count,
        "seed": args.seed,
        "datasets": {},
    }
    for name, info in DATASETS.items():
        root = args.data_root / name
        panel = choose_panel(root, name, info, args.sample_count, args.seed)
        payload["datasets"][name] = panel
        print(f"{panel['dataset']}: {panel['sample_count']} items across {panel['class_count']} classes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote fixed panel manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
