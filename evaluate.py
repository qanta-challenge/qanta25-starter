import argparse
import glob
import json
import os
from collections import defaultdict
from typing import TypedDict

from datasets import load_dataset
from rich import print as rprint

from io_utils import read_jsonl
from metrics import compute_bonus_metrics, compute_tossup_metrics
from metrics.qb_metrics import (
    TossupModelOutput,
    TossupRunOutput,
    compute_tossup_metrics,
)


def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_output_files(outputs_dir="outputs"):
    """
    Returns a dict: {dataset: {mode: {model: filepath}}}
    """
    results = defaultdict(lambda: defaultdict(dict))
    for dataset in os.listdir(outputs_dir):
        dataset_dir = os.path.join(outputs_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for mode in os.listdir(dataset_dir):
            mode_dir = os.path.join(dataset_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for file in glob.glob(os.path.join(mode_dir, "*.jsonl")):
                model = os.path.splitext(os.path.basename(file))[0]
                results[dataset][mode][model] = file
    return results


def main():
    outputs = find_output_files()
    summary = []

    for dataset, modes in outputs.items():
        for mode, models in modes.items():
            for model, filepath in models.items():
                print(f"Evaluating {dataset}/{mode}/{model} ...")
                predictions = load_jsonl(filepath)
                # You may want to load the gold data here for more advanced metrics
                # For now, assume predictions contain all necessary info
                if mode == "bonus":
                    metrics = evaluate_bonus_metrics(predictions)
                elif mode == "tossup":
                    metrics = evaluate_tossup_model(predictions)
                else:
                    print(f"Unknown mode: {mode}, skipping.")
                    continue
                summary.append(
                    {"dataset": dataset, "mode": mode, "model": model, **metrics}
                )

    # Print summary table
    if summary:
        keys = list(summary[0].keys())
        print("\nEvaluation Summary:")
        print("\t".join(keys))
        for row in summary:
            print("\t".join(str(row[k]) for k in keys))
    else:
        print("No results found in outputs/.")


def evaluate_tossup_model(model_name, dataset_name, examples_by_qid):
    filepath = f"outputs/{dataset_name}/tossup/{model_name}.jsonl"

    records = read_jsonl(filepath)

    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)

    system_outputs: list[TossupModelOutput] = []
    for qid, records in grouped.items():
        run_outputs: list[TossupRunOutput] = []
        scores = []
        for rec in records:
            run_out = {
                "buzz": rec["buzz"],
                "correct": bool(rec["correct"]),
                "token_position": rec.get("token_position", None),
            }
            if run_out["token_position"] is None:
                run_index = rec["run_number"] - 1
                run_out["token_position"] = (
                    examples_by_qid[qid]["run_indices"][run_index] + 1
                )
            run_outputs.append(run_out)
        run_outputs.sort(key=lambda x: x["token_position"])
        system_outputs.append(
            {"qid": str(qid), "run_outputs": run_outputs, "scores": scores}
        )

    qids = [e["qid"] for e in system_outputs]
    run_indices = [examples_by_qid[qid]["run_indices"] for qid in qids]
    human_buzz_points = [
        examples_by_qid[qid]["human_buzz_positions"] for qid in qids
    ]  # Assuming this is a list of lists

    return compute_tossup_metrics(system_outputs, run_indices, human_buzz_points)


def evaluate_bonus_metrics(model_name, dataset_name, examples_by_qid):
    filepath = f"outputs/{dataset_name}/bonus/{model_name}.jsonl"
    records = read_jsonl(filepath)

    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)

    system_outputs = []
    for qid, records in grouped.items():
        part_outputs = []
        part_scores = []
        for rec in records:
            part_output = {
                "part_number": rec["part_number"],
                "correct": rec["correct"],
            }
            part_outputs.append(part_output)
        system_outputs.append(
            {"qid": str(qid), "part_outputs": part_outputs, "scores": part_scores}
        )

    return compute_bonus_metrics(system_outputs)


def evaluate_model(model_name, dataset_name, mode, examples_by_qid):
    if mode == "bonus":
        return evaluate_bonus_metrics(model_name, dataset_name, examples_by_qid)
    elif mode == "tossup":
        return evaluate_tossup_model(model_name, dataset_name, examples_by_qid)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_parser():

    parser = argparse.ArgumentParser(description="Evaluate Tossup Model")
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
        "--task",
        "-c",
        type=str,
        required=True,
        choices=["bonus", "tossup"],
        help="Task.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Name of the HF model to use for the pipeline.",
    )
    return parser


if __name__ == "__main__":

    from tabulate import tabulate

    parser = create_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.mode, split="eval")
    examples_by_qid = {str(e["qid"]): e for e in dataset}
    dataset_name = args.dataset.split("/")[-1]

    import pandas as pd

    if args.model is None:
        print(
            f"Model not specified, using all model outputs found for {dataset_name}:{args.mode}"
        )
        model_names = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(f"outputs/{dataset_name}/{args.mode}/*.jsonl")
        ]
        df_list = []
        for model_name in model_names:
            print(f"Evaluating {model_name}...")
            metrics = evaluate_model(
                model_name, dataset_name, args.mode, examples_by_qid
            )
            metrics["model"] = model_name
            df_list.append(metrics)
        df = pd.DataFrame(df_list)
    else:
        model_name = args.model.split("/")[-1]
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model_name, dataset_name, args.mode, examples_by_qid)
        metrics["model"] = model_name
        df = pd.DataFrame([metrics])

    # remove "human_win_rate_strict"
    if "human_win_rate_strict" in df.columns:
        df = df.drop(columns=["human_win_rate_strict"])

    # Move the model column to the front
    model_col = df.pop("model")
    df.insert(0, "model", model_col)
    # remove tossup score
    if "tossup_score" in df.columns:
        df = df.drop(columns=["tossup_score"])
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
