import json
import time


# IO utils
def write_jsonl(data: dict, filepath, mode="w"):
    """Write a list of dictionaries to a JSONL file."""
    with open(filepath, mode) as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=True)
            f.write(f"{json_str}\n")


def write_entry_to_jsonl(data, filepath):
    """Write a list of dictionaries to a JSONL file."""
    write_jsonl([data], filepath, mode="a")


def read_jsonl(filepath):
    """Read a JSONL file and return a list of dictionaries."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def dual_percent_bar(
    n_total,
    n_completed,
    n_in_progress,
    postfix_str: str = "",
    bar_length: int = 30,
    start_time: float = None,
) -> None:
    # Unicode block characters for smoother progress bar
    completed_pct = n_completed / n_total if n_total else 0
    in_progress_pct = n_in_progress / n_total if n_total else 0
    completed_blocks = int(bar_length * completed_pct)
    in_progress_blocks = int(bar_length * in_progress_pct)
    not_started_blocks = bar_length - completed_blocks - in_progress_blocks

    # Use Unicode blocks for a smoother look
    char_C = "█"
    char_P = "▓"
    char_N = "░"

    bar = ""
    if completed_blocks > 0:
        bar += "\033[92m" + char_C * completed_blocks  # Green for completed
    if in_progress_blocks > 0:
        bar += "\033[93m" + char_P * in_progress_blocks  # Yellow for in progress
    if not_started_blocks > 0:
        bar += "\033[0m" + char_N * not_started_blocks  # Reset for not started
    bar += "\033[0m"  # Ensure reset at end

    # Time estimate
    time_str = ""
    if start_time is not None and n_completed > 0:
        elapsed = time.time() - start_time
        rate = elapsed / n_completed
        remaining = n_total - n_completed
        eta = remaining * rate
        mins, secs = divmod(int(eta), 60)
        time_str = f" | ETA: {mins}m{secs:02d}s"
    elif start_time is not None:
        time_str = " | ETA: --"

    print(
        f"\r│{bar}│ {completed_pct:.1%} complete, {in_progress_pct:.1%} in progress │ "
        f"{postfix_str}{time_str}",
        end="",
    )
