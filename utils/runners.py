import os
import time

import numpy as np
from tqdm.auto import tqdm

from dataloaders import BonusDatasetIterator, TossupDatasetIterator
from io_utils import dual_percent_bar, write_entry_to_jsonl
from metrics import evaluate_prediction


def evaluate_output(output, answers):
    """
    Evaluate the output against the example and return the guess and correctness.
    """
    output["guess"] = output.pop("answer")
    output["correct"] = evaluate_prediction(output["guess"], answers)


# The main pipeline runner
def run_bonus_pipeline(pipe, dataset_iterator: BonusDatasetIterator, filepath):
    """
    Run the pipeline on the dataset and save the outputs to a JSONL file.
    """
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    scores = []
    with tqdm(dataset_iterator, desc="Processing examples") as pbar:
        for batch in pbar:
            output_batch = pipe(batch)
            for i, output in enumerate(output_batch):
                # Evaluate the output and write to file
                evaluate_output(output, batch["clean_answers"][i])
                output["qid"] = batch["qid"][i]
                output["id"] = batch["id"][i]
                output["part_number"] = batch["part_number"][i]
                scores.append(output["correct"])
                write_entry_to_jsonl(output, filepath)
            acc = np.mean(scores)
            pbar.set_postfix_str(f"Accuracy: {acc:.1%}")


def run_tossup_pipeline(pipe, dataset_iterator: TossupDatasetIterator, filepath):
    """
    Run the pipeline on the dataset and save the outputs to a JSONL file.
    """
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    scores = []
    start_time = time.time()
    dual_percent_bar(
        len(dataset_iterator),
        dataset_iterator.n_completed,
        dataset_iterator.n_in_progress,
        start_time=start_time,
    )
    for batch in dataset_iterator:
        output_batch = pipe(batch)
        for i, output in enumerate(output_batch):
            # Evaluate the output and write to file
            evaluate_output(output, batch["clean_answers"][i])
            output["qid"] = batch["qid"][i]
            output["id"] = batch["id"][i]
            output["run_number"] = batch["run_number"][i]
            output["token_position"] = batch["token_position"][i]
            scores.append(output["correct"])
            write_entry_to_jsonl(output, filepath)
        dataset_iterator.update(batch, output_batch)
        acc = np.mean(scores)
        avg_run_number = np.mean(list(dataset_iterator.curr_runs.values()))
        dual_percent_bar(
            len(dataset_iterator),
            dataset_iterator.n_completed,
            dataset_iterator.n_in_progress,
            postfix_str=f"Accuracy: {acc:.1%} | Avg Run Number: {avg_run_number:.1f}",
            start_time=start_time,
        )
