"""
🚀 QANTA Pipeline Runner

This script runs custom Hugging Face pipelines for Quiz Bowl question answering tasks.
It supports both "bonus" and "tossup" modes, with flexible dataset processing and batch handling.

FEATURES:
- ✅ Automatic pipeline discovery and registration
- ✅ Batch processing with configurable batch sizes
- ✅ Resume functionality (skip already processed examples)
- ✅ Debug mode for development
- ✅ Progress tracking and logging

USAGE EXAMPLES:

1. Run a bonus pipeline:
   python run_pipeline.py pipelines.my_bonus_pipeline \
     --model Qwen/Qwen2.5-3B-Instruct \
     --mode bonus \
     --batch_size 8

2. Run a tossup pipeline:
    python run_pipeline.py pipelines.my_tossup_pipeline \
     --model Qwen/Qwen2.5-3B-Instruct \
     --mode tossup \
     --batch_size 8

3. Debug mode (limits to 3 examples): This is useful for development.
   python run_pipeline.py pipelines.my_bonus_pipeline \
     --model Qwen/Qwen2.5-3B-Instruct \
     --mode bonus \
     --debug

4. Reprocess existing outputs:
   python run_pipeline.py pipelines.my_bonus_pipeline \
     --model Qwen/Qwen2.5-3B-Instruct \
     --mode bonus \
     --reprocess

OUTPUT FORMAT:
- Results are saved to: outputs/{dataset_name}/{mode}/{model_name}.jsonl
- Each line contains the model's prediction for one example
- Resume functionality automatically skips already processed examples

REQUIREMENTS:
- Your pipeline must be in the pipelines/ directory
- Pipeline class must be properly registered with transformers
- Model must be compatible with the specified task type

For more information, see the README.md file.
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm
from transformers import pipeline

from dataloaders import BonusDatasetIterator, TossupDatasetIterator
from io_utils import read_jsonl
from runners import run_bonus_pipeline, run_tossup_pipeline


# Dataset preparation utils
def parse_packets(packet_str: str) -> list[int]:
    """
    Parse a packet string into a list of integers.
    Supports:
      - Single packet: "12"
      - Range: "4-5"
      - Comma separated: "2,4,5"
      - Mixed: "2,4-6,9"
    """
    if not packet_str:
        return []
    packet_str = packet_str.strip()
    packets = set()
    for part in packet_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-")
                packets.update(range(int(start), int(end) + 1))
            except ValueError:
                raise ValueError(f"Invalid range in packet string: '{part}'")
        else:
            try:
                packets.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid packet number: '{part}'")
    return sorted(packets)


def prepare_dataset(dataset_name, mode, packets=None, force_redownload=False):
    """
    Load and prepare the dataset for processing.
    """
    download_mode = "force_redownload" if force_redownload else None
    dataset = load_dataset(
        dataset_name, mode, split="eval", download_mode=download_mode
    )

    # Filtering the dataset if packets are specified
    def get_packet_number(qid):
        if isinstance(qid, str) and "-" in qid:
            return qid.split("-")[-2]
        return int(qid) // 20 + 1

    if packets:
        p_set = set(parse_packets(packets))
        dataset = dataset.filter(lambda x: get_packet_number(x["qid"]) in p_set)

    return dataset


def create_dataset_iterator(dataset, mode, batch_size=4):
    if mode == "bonus":
        return BonusDatasetIterator(dataset, batch_size)
    elif mode == "tossup":
        return TossupDatasetIterator(dataset, batch_size)
    raise ValueError(f"Unsupported mode: {mode}")


def create_dummy_pipeline(config: str):
    if config == "bonus":
        import time

        def dummy_pipeline(batch):
            bs = len(batch["qid"])
            return [
                {"answer": "Answer", "confidence": 0.4, "explanation": "hahaha"}
                for _ in range(bs)
            ]
            time.sleep(0.5)

        return dummy_pipeline
    elif config == "tossup":
        import random

        def dummy_pipeline(batch):
            bs = len(batch["qid"])
            return [
                {"answer": "Answer", "confidence": 0.4, "buzz": random.random() < 0.2}
                for _ in range(bs)
            ]
            time.sleep(1.0)

        return dummy_pipeline
    raise ValueError(f"Unsupported config: {config}")


def main(args: argparse.Namespace):
    # Setting up the output directory
    ds_dirname = args.dataset.split("/")[1]
    output_dir = f"outputs/{ds_dirname}/{args.mode}"
    submission_id = args.model.split("/")[-1].lower()

    # Load the dataset
    dataset = prepare_dataset(
        args.dataset, args.mode, args.packets, args.force_redownload
    )
    if args.debug:
        logger.info("Debug mode is enabled, will limit to 3 examples.")
        dataset = dataset.select(range(3))  # Limit to 10 examples for debugging
        submission_id += "debug"
        args.batch_size = 1

    print(f"Dataset: {args.dataset}, Config: {args.mode}, Packets: {args.packets}")
    print(dataset)

    filepath = f"{output_dir}/{submission_id}.jsonl"

    dataset_iterator = create_dataset_iterator(dataset, args.mode, args.batch_size)
    if args.reprocess:
        logger.info("Reprocessing is enabled, will overwrite existing outputs.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        open(filepath, "w").close()  # Clear the file if it exists

    elif os.path.exists(filepath):
        system_outputs = read_jsonl(filepath)
        dataset_iterator.mark_processed(system_outputs)
        logger.info(
            f"Filtered dataset to only include unprocessed examples: {len(dataset)} remaining."
        )

    # pipe = create_dummy_pipeline(args.mode)
    pipe = pipeline(
        f"quizbowl-{args.mode}",
        model=args.model,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.mode == "bonus":
        run_bonus_pipeline(pipe, dataset_iterator, filepath)
    else:
        run_tossup_pipeline(pipe, dataset_iterator, filepath)
    print(f"Outputs saved to {filepath}")


def add_arguments():
    parser = argparse.ArgumentParser(description="Run the bonus pipeline for Qanta.")
    parser.add_argument(
        "pipeline_module",
        type=str,
        help="The class of the pipeline to run, e.g., 'QwenBonusPipeline'.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="qanta-challenge/qanta25-playground",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--mode",
        "--config",
        "-c",
        type=str,
        default="bonus",
        choices=["bonus", "tossup"],
        help="Mode of the pipeline.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the HF model to use for the pipeline.",
    )
    parser.add_argument(
        "--packets",
        type=str,
        default="",
        help="Packet numbers to process, e.g., '1-5' or '2,4,6' or '1-4,7,10'.",
    )
    parser.add_argument(
        "--reprocess",
        "-r",
        action="store_true",
        help="Reprocess the dataset even if outputs already exist.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=4,
        help="Batch size for processing examples.",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Force redownload of the dataset.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose output.",
    )
    return parser


if __name__ == "__main__":
    # Import your pipeline class here before running the script:

    # Load registered pipeline modules dynamically
    registered = set()
    for root, dirs, files in os.walk("pipelines"):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__"):
                rel_path = os.path.relpath(os.path.join(root, filename), "pipelines")
                module_name = rel_path[:-3].replace(os.sep, ".")
                registered.add(f"pipelines.{module_name}")

    print("📦 Registered pipeline modules:")
    for module in sorted(registered):
        print(f"  - {module}")
    import importlib

    parser = add_arguments()
    args = parser.parse_args()
    # if the module is provided in file path format, import it
    # if "/" in args.pipeline_module:
    #     module_name, class_name = args.pipeline_module.rsplit("/", 1)
    #     args.pipeline_module = f"{module_name}.{class_name}"
    # else:
    #     class_name = args.pipeline_module
    pipeline_module = args.pipeline_module.replace("/", ".").removesuffix(".py")
    if not pipeline_module:
        raise ValueError("Pipeline module must be specified.")
    if f"pipelines.{pipeline_module}" in registered:
        pipeline_module = f"pipelines.{pipeline_module}"
    # Import the pipeline module dynamically
    print(f"⤵️ Importing pipeline module: {pipeline_module} 📦")
    importlib.import_module(pipeline_module)
    main(args)
    print("Done!")
