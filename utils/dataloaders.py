# %%
import heapq

from datasets import Dataset
from loguru import logger


def extract_run_example(example: dict, run_index: int) -> dict:
    """
    Extract a single run example from a list of examples.
    """
    token_position = example["run_indices"][run_index]
    return {
        "qid": example["qid"],
        "id": f"{example['qid']}_{run_index + 1:02d}",
        "run_number": run_index + 1,
        "token_position": token_position,
        "question_text": " ".join(example["question"].split()[:token_position]),
        "clean_answers": example["clean_answers"],
    }


def flatten_parts(batch: list[dict]):
    # each element in the incoming batch has a single qid/question
    # but a *list* of parts; we turn that into one entry per part
    flat_batch = {
        "id": [],
        "qid": [],
        "part_number": [],
        "leadin": [],
        "part": [],
        "clean_answers": [],
    }

    for qid, leadin, parts in zip(batch["qid"], batch["leadin"], batch["parts"]):
        n_parts = len(parts)
        flat_batch["qid"].extend([qid] * n_parts)
        flat_batch["leadin"].extend([leadin] * n_parts)
        for part in parts:
            flat_batch["id"].append(f"{qid}_{part['number']:02d}")
            flat_batch["part_number"].append(part["number"])
            flat_batch["part"].append(part["question"])
            flat_batch["clean_answers"].append(part["clean_answers"])

    return flat_batch


class TossupDatasetIterator:
    def __init__(self, dataset: Dataset, batch_size: int = 4):
        self.batch_size = batch_size
        self.examples = {qid: dataset[qid] for qid in dataset["qid"]}

        # number of run_indices for each example (qid)
        self.num_runs = {
            qid: len(run_indices)
            for qid, run_indices in zip(dataset["qid"], dataset["run_indices"])
        }
        # 0-indexed current index of the run_indices list for each example (qid)
        self.curr_runs = dict.fromkeys(dataset["qid"], 0)
        self._build_run_heap()
        self.n_completes = []

    def _push(self, qid: str):
        heapq.heappush(self._run_heap, (-self.curr_runs[qid], qid))

    def _pop(self):
        if not self._run_heap:
            raise IndexError("Heap is empty")
        priority, qid = heapq.heappop(self._run_heap)
        return -priority, qid

    def _build_run_heap(self):
        self._run_heap = [(-self.curr_runs[qid], qid) for qid in self.curr_runs]
        heapq.heapify(self._run_heap)

    @property
    def n_completed(self):
        return len(self.examples) - len(self.curr_runs)

    @property
    def n_in_progress(self):
        return sum(int(i > 0) for i in self.curr_runs.values())

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.examples)

    def __next__(self):
        batch = []
        n_completed = 0
        while self._run_heap and len(batch) < self.batch_size:
            run_index, qid = self._pop()

            if qid not in self.curr_runs or run_index != self.curr_runs[qid]:
                # qid has been exhausted or the run_index is outdated
                continue

            batch.append(extract_run_example(self.examples[qid], run_index))
            if self.curr_runs[qid] == self.num_runs[qid] - 1:
                # This is the last run of this qid, remove from curr_runs
                del self.curr_runs[qid]
                n_completed += 1
            # else:
            #     self.curr_runs[qid] += 1
            #     self._push(qid)

        for e in batch:
            if e["qid"] in self.curr_runs:
                self.curr_runs[e["qid"]] += 1
                self._push(e["qid"])

        if len(batch) == 0:
            # Clean up the heap for the next iteration
            self._run_heap = []
            raise StopIteration
        # convert list of dicts to dict of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        self.n_completes.append(n_completed)
        return batch

    def update(self, examples, outputs: list[dict]):
        for qid, output in zip(examples["qid"], outputs):
            if (
                output["buzz"]
                and qid in self.curr_runs
                and self.curr_runs[qid] < self.num_runs[qid] - 1
            ):
                # If the model buzzed and there are more runs for this qid,
                # evaluate the last run of this qid
                self.curr_runs[qid] = self.num_runs[qid] - 1
                self._push(qid)

    def mark_processed(self, outputs: list[dict]):
        """This function is called only once after initialization in order to filter out examples that have already been processed."""
        # Prepare the highest run number (1-indexed) for each qid
        highest_runs = {}
        for output in outputs:
            qid = output["qid"]
            run_number = output["run_number"]
            highest_runs[qid] = max(highest_runs.get(qid, 0), run_number)

        for qid, highest_run in highest_runs.items():
            if qid in self.curr_runs and highest_run == self.num_runs[qid]:
                # If the run index is higher than the current run, remove it
                del self.curr_runs[qid]
            else:
                # Otherwise, update the current run index
                self.curr_runs[qid] = highest_run
        self._build_run_heap()


class BonusDatasetIterator:
    def __init__(self, dataset: Dataset, batch_size: int = 4):
        self.dataset = dataset.map(
            flatten_parts, batched=True, remove_columns=dataset.column_names
        )
        self.batch_size = batch_size
        self.batched_ds = self.dataset.batch(self.batch_size)

    def mark_processed(self, outputs: list[dict]):
        output_ids = {output["id"] for output in outputs}
        self.dataset = self.dataset.filter(lambda x: x["id"] not in output_ids)
        self.batched_ds = self.dataset.batch(self.batch_size)

    def __iter__(self):
        return iter(self.batched_ds)

    def __next__(self):
        return next(self.batched_ds)

    def __len__(self):
        return len(self.batched_ds)


if __name__ == "__main__":
    # Test the iterator
    import random
    import time

    from datasets import load_dataset
    from tqdm.auto import tqdm

    from io_utils import dual_percent_bar

    def dummy_pipeline(batch):
        bs = len(batch["qid"])
        return [
            {"qid": batch["qid"][i], "buzz": random.random() < 0.2} for i in range(bs)
        ]

    dataset = load_dataset("qanta-challenge/qanta25-playground", "tossup", split="eval")
    dataset = dataset.select(range(20))
    iterator = TossupDatasetIterator(dataset, batch_size=8)
    n_iters = 0
    print("len it:", len(iterator))
    all_outputs = []
    for batch in iterator:
        time.sleep(0.5)
        n_iters += 1
        outputs = dummy_pipeline(batch)
        for idx, out in zip(batch["id"], outputs):
            out["id"] = idx
        all_outputs.extend(outputs)
        iterator.update(batch, outputs)
        dual_percent_bar(len(iterator), iterator.n_completed, iterator.n_in_progress)
    print(
        f"# Iterations for {len(dataset)} examples with batch size {iterator.batch_size}: {n_iters}"
    )

    dataset = load_dataset("qanta-challenge/qanta25-playground", "bonus", split="eval")
    dataset = dataset.select(range(20))
    b_ds = BonusDatasetIterator(dataset, batch_size=4)
    for batch in tqdm(b_ds):
        # print(batch)
        time.sleep(0.05)

    ids_by_qid = {}
    for out in all_outputs:
        idx = out["id"]
        qid, run_index = idx.split("_")
        ids_by_qid.setdefault(qid, []).append([int(run_index), out["buzz"]])
    ids_by_qid = {k: sorted(v) for k, v in ids_by_qid.items()}
    for idx, buzzes in ids_by_qid.items():
        assert sum(b[1] for b in buzzes) <= 2, f"Multiple buzzes for {idx}: {buzzes}"
    ids_by_qid
