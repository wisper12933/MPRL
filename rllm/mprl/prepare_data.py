from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprl.task_specs import DEFAULT_ALFWORLD_DATA, TASK_SPECS, load_task_dataset


def prepare_data(
    task: str,
    *,
    alfworld_data: str | Path = DEFAULT_ALFWORLD_DATA,
    train_limit: int | None = None,
    test_limit: int | None = None,
):
    """Load normalized task dictionaries directly from the maintained splits."""
    train_dataset = load_task_dataset(task, "train", alfworld_data=alfworld_data, limit=train_limit)
    test_dataset = load_task_dataset(task, "test", alfworld_data=alfworld_data, limit=test_limit)
    return train_dataset, test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalized MPRL train/test task splits.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--alfworld-data", default=str(DEFAULT_ALFWORLD_DATA))
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_data(
        args.task,
        alfworld_data=args.alfworld_data,
        train_limit=args.limit,
        test_limit=args.limit,
    )
    print(
        json.dumps(
            {
                "task": args.task,
                "train_count": len(train_dataset),
                "test_count": len(test_dataset),
                "train_sample": train_dataset[0] if len(train_dataset) else None,
                "test_sample": test_dataset[0] if len(test_dataset) else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
