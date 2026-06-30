"""
Load and preview the MT-Bench human-judgment data from HuggingFace.

Run this before anything else: it downloads the dataset and caches a local
copy at data/mt_bench.csv, which every other script reads from.
"""

import os

from datasets import load_dataset
import pandas as pd


def load_mt_bench():
    ds = load_dataset("lmsys/mt_bench_human_judgments")
    print(f"\nDataset splits: {list(ds.keys())}")
    return ds


def explore(ds):
    df = pd.DataFrame(ds["human"])
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nWinner distribution:\n{df['winner'].value_counts()}")
    print(f"\nJudges:\n{df['judge'].value_counts()}")
    print(f"\n--- Sample row ---")
    row = df.iloc[0]
    print(f"Question ID: {row['question_id']}")
    print(f"Model A: {row['model_a']}")
    print(f"Model B: {row['model_b']}")
    print(f"Winner: {row['winner']}")
    print(f"Turn: {row['turn']}")
    print(f"Conversation A (truncated): {str(row['conversation_a'])[:300]}")
    return df


if __name__ == "__main__":
    ds = load_mt_bench()
    df = explore(ds)
    # Cache a local copy so downstream scripts don't re-download the dataset.
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/mt_bench.csv", index=False)
    print("\nSaved to data/mt_bench.csv")
