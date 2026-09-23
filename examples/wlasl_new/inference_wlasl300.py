#!/usr/bin/env python3
"""Evaluate WLASL300 test poses and export every prediction and exact matches.

Run with the existing Uni-Sign environment, or submit
``sbatch wlasl300_finetuning/run_inference.sh`` from the workspace root.
Missing poses are listed in summary.json and excluded from accuracy.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE = REPO_ROOT.parent
CSV_FIELDS = ("video_id", "ground_truth", "gloss_index", "prediction", "is_correct", "video_path")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=WORKSPACE / "wlasl300_finetuning/best_checkpoint.pth")
    parser.add_argument("--data-config", type=Path,
                        default=REPO_ROOT / "configs/wlasl300_test.json")
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "wlasl300_finetuning/inference")
    parser.add_argument("--video-root", type=Path,
                        default=Path("/pfs/lustrep2/scratch/project_465002625/data/WLASL300/all_videos"))
    parser.add_argument("--gloss-index-file", type=Path,
                        default=WORKSPACE / "wlasl300_finetuning/gloss_to_index.json",
                        help="JSON mapping of ground-truth glosses to test-directory indices")
    parser.add_argument("--mt5-path", type=Path, default=REPO_ROOT / "pretrained_weight/mt5-base")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bf16", "float32"), default="auto",
                        help="auto uses BF16 on GPU, matching training, and FP32 on CPU")
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--num-beams", type=positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=positive_int,
                        help="optional smoke-test sample limit; omitted means all usable test videos")
    return parser


def read_gloss_indices(path):
    with Path(path).open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Gloss-index mapping must be a non-empty JSON object.")
    for gloss, index in mapping.items():
        if not isinstance(gloss, str) or not gloss or type(index) is not int or index < 0:
            raise ValueError(f"Invalid gloss-index entry: {gloss!r}: {index!r}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Gloss indices must be unique.")
    return mapping


def read_references(annotation_path):
    with Path(annotation_path).open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    references = {}
    for video in annotation.values():
        for clip_id in video["clip_order"]:
            if clip_id in references:
                raise ValueError(f"Duplicate video ID in test annotation: {clip_id}")
            label = video[clip_id]["translation"]
            if not isinstance(label, str) or not label:
                raise ValueError(f"Missing ground truth for {clip_id}")
            references[clip_id] = label
    return references


def prediction_row(video_id, ground_truth, prediction, video_root, gloss_index):
    # Use exactly the same equality rule as SLRT_metrics.islr_performance.
    return {
        "video_id": video_id,
        "ground_truth": ground_truth,
        "gloss_index": gloss_index,
        "prediction": prediction,
        "is_correct": int(prediction == ground_truth),
        "video_path": os.path.relpath(
            (Path(video_root) / f"{video_id}.mp4").resolve(), WORKSPACE
        ),
    }


def export_results(rows, output_dir, metadata):
    """Write both complete CSVs, including headers when there are no matches."""
    if not rows:
        raise ValueError("No predictions were produced.")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate video predictions; refusing to export misleading results.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correct_rows = [row for row in rows if row["is_correct"]]
    for filename, selected in (("all_predictions.csv", rows),
                               ("true_positives.csv", correct_rows)):
        path = output_dir / filename
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(selected)
        temporary.replace(path)

    class_totals = Counter(row["ground_truth"] for row in rows)
    class_correct = Counter(row["ground_truth"] for row in correct_rows)
    summary = dict(metadata)
    summary.update({
        "tested_videos": len(rows),
        "correct_predictions": len(correct_rows),
        "top1_accuracy_percent": 100 * len(correct_rows) / len(rows),
        "mean_per_class_accuracy_percent": 100 * sum(
            class_correct[label] / count for label, count in class_totals.items()
        ) / len(class_totals),
        "matching_rule": "Exact string equality, matching SLRT_metrics.islr_performance",
    })
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "summary.json")
    return summary


def main(args):
    # Lazy imports allow --help and CSV tests without the GPU environment.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from data_config import load_data_config, get_required_split_specs, preflight_data_config
    from datasets import ConfiguredDataset
    from models import Uni_Sign

    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU is available. Submit run_inference.sh on LUMI, or use --device cpu.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (args.dtype == "bf16" or
                              (args.dtype == "auto" and device.type == "cuda")) else torch.float32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_data_config(args.data_config)
    specs = get_required_split_specs(config, "test")
    if len(specs) != 1 or specs[0].get("loader") != "ytasl_json":
        raise ValueError("WLASL300 inference expects one ytasl_json test dataset.")
    report = preflight_data_config(config, active_splits=("test",))
    if report["errors"]:
        raise ValueError("\n".join(report["errors"]))
    if report["rgb_requested"]:
        raise ValueError("The WLASL300 checkpoint uses pose inputs only.")

    # Architecture and preprocessing from 0001-train_unisign_wlasl300.sh.
    model_args = SimpleNamespace(
        layout=config["layout"], graph=config["graph"],
        target_language=config["target_language"], normalization=config.get("normalization", "none"),
        rgb_support=False, task="ISLR", hidden_dim=256, no_adaptive_gcn=False,
        n_registers=0, register_position="before_all", label_smoothing=0.2,
        mt5_path=str(args.mt5_path), max_length=args.max_length,
        normalize_text=False, clip_merge_mode="none",
    )
    dataset = ConfiguredDataset(specs=specs, args=model_args, phase="test")
    references = read_references(specs[0]["annotation_path"])
    gloss_indices = read_gloss_indices(args.gloss_index_file)
    unmapped = sorted(set(references.values()) - set(gloss_indices))
    if unmapped:
        raise ValueError(f"Ground-truth glosses missing from index mapping: {unmapped}")
    missing = set(dataset.loaders[0].missing_clip_names)
    available_ids = [name for name in references if name not in missing]
    if len(dataset) != len(available_ids) or not available_ids:
        raise ValueError("Test annotation and usable dataset counts do not agree, or no poses are available.")
    expected_ids = available_ids[:args.limit] if args.limit else available_ids
    selected = Subset(dataset, range(len(expected_ids))) if args.limit else dataset
    print(f"Test videos: {len(references)}; usable: {len(dataset)}; missing poses: {len(missing)}", flush=True)
    if missing:
        print("Missing pose IDs: " + ", ".join(sorted(missing)), flush=True)
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate_fn, pin_memory=device.type == "cuda")

    print(f"Loading {args.checkpoint} on {device} ({dtype})", flush=True)
    model = Uni_Sign(args=model_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device=device, dtype=dtype).eval()

    rows = []
    with torch.inference_mode():
        for src_input, tgt_input in loader:
            for key, value in src_input.items():
                if isinstance(value, torch.Tensor):
                    src_input[key] = value.to(device=device,
                                              dtype=dtype if value.is_floating_point() else value.dtype,
                                              non_blocking=device.type == "cuda")
            # Match the repository evaluator. generate consumes pose embeddings
            # and the attention mask; the teacher-forced logits/loss are unused.
            encoded = model(src_input, tgt_input)
            token_ids = model.generate(encoded, max_new_tokens=args.max_new_tokens,
                                       num_beams=args.num_beams)
            predictions = model.mt5_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            names = src_input["name_batch"]
            if len(predictions) != len(names):
                raise RuntimeError("Prediction count does not match the input batch.")
            for name, prediction in zip(names, predictions):
                rows.append(prediction_row(
                    name, references[name], prediction, args.video_root,
                    gloss_indices[references[name]],
                ))
            print(f"Predicted {len(rows)}/{len(expected_ids)} videos", flush=True)

    if [row["video_id"] for row in rows] != expected_ids:
        raise RuntimeError("Predictions do not cover the requested test videos exactly once in order.")
    summary = export_results(rows, args.output_dir, {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_config": str(args.data_config.resolve()),
        "gloss_index_file": str(args.gloss_index_file.resolve()),
        "annotation_path": specs[0]["annotation_path"],
        "annotated_videos": len(references),
        "available_pose_videos": len(available_ids),
        "missing_pose_videos": [{"video_id": name, "ground_truth": references[name]} for name in sorted(missing)],
        "device": str(device), "dtype": str(dtype), "seed": args.seed,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "sample_limit": args.limit,
    })
    print(f"Correct: {summary['correct_predictions']}/{summary['tested_videos']} "
          f"({summary['top1_accuracy_percent']:.2f}%)", flush=True)
    print(f"Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(get_parser().parse_args())
