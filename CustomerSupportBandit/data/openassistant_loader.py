"""
OpenAssistant Conversations dataset loader.

Loads ~161K messages with human quality ratings and rank annotations.
Used for reward model calibration and quality signal validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_PATHS


def load_openassistant_data(use_parquet: bool = True,
                            sample_frac: Optional[float] = None,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Load the OpenAssistant Conversations dataset.

    Parameters
    ----------
    use_parquet : bool
        If True, load from parquet files (preferred — has structured labels).
        If False, fall back to CSV.
    sample_frac : float, optional
        Fraction to sample.

    Returns
    -------
    pd.DataFrame
        Combined train + val data with extracted label columns.
    """
    if use_parquet:
        train_path = DATA_PATHS.get("openassistant_train")
        val_path = DATA_PATHS.get("openassistant_val")

        if train_path and Path(train_path).exists():
            print(f"Loading OpenAssistant from parquet ...")
            df_train = pd.read_parquet(train_path, engine='fastparquet')
            df_val = pd.read_parquet(val_path, engine='fastparquet')
        else:
            print("Parquet not found, falling back to CSV ...")
            use_parquet = False

    if not use_parquet:
        train_path = DATA_PATHS.get("openassistant_train_csv")
        val_path = DATA_PATHS.get("openassistant_val_csv")
        print(f"Loading OpenAssistant from CSV ...")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)

    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df = pd.concat([df_train, df_val], ignore_index=True)

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    print(f"  Loaded {len(df):,} messages (train={len(df_train):,}, val={len(df_val):,})")

    # Extract structured labels if parquet
    if use_parquet and 'labels.name' in df.columns:
        df = _extract_labels(df)

    # Rename detoxify columns
    rename_map = {c: 'detox_' + c.split('.', 1)[1]
                  for c in df.columns if c.startswith('detoxify.')}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return df


def _extract_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Extract label columns from labels.name / labels.value lists."""
    records = []
    for names, values in zip(df['labels.name'], df['labels.value']):
        if isinstance(names, list) and isinstance(values, list):
            records.append(dict(zip(names, values)))
        else:
            records.append({})
    label_df = pd.DataFrame(records)

    for col in label_df.columns:
        df[f'label_{col}'] = label_df[col].values

    n_quality = df.get('label_quality', pd.Series(dtype=float)).notna().sum()
    print(f"  Extracted labels: {list(label_df.columns)}")
    print(f"  Quality labels available: {n_quality:,} / {len(df):,}")
    return df


def build_conversation_trees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct conversation trees from parent_id chains.

    Returns DataFrame with one row per conversation tree:
    - tree_id: message_id of root
    - messages: list of message dicts in tree order
    - num_turns: number of messages
    - avg_quality: mean quality score across rated messages
    - roles: sequence of roles (prompter/assistant)
    """
    print("Building conversation trees ...")

    # Find roots (messages with no parent)
    roots = df[df['parent_id'].isna()]['message_id'].values

    # Build children map
    children_map = {}
    for _, row in df.iterrows():
        pid = row.get('parent_id')
        if pd.notna(pid):
            if pid not in children_map:
                children_map[pid] = []
            children_map[pid].append(row['message_id'])

    msg_lookup = df.set_index('message_id')

    trees = []
    for root_id in roots:
        messages = []
        queue = [root_id]
        while queue:
            mid = queue.pop(0)
            if mid in msg_lookup.index:
                row = msg_lookup.loc[mid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                messages.append({
                    'message_id': mid,
                    'text': str(row.get('text', '')),
                    'role': row.get('role', 'unknown'),
                    'rank': row.get('rank', np.nan),
                    'quality': row.get('label_quality', np.nan),
                })
            if mid in children_map:
                queue.extend(children_map[mid])

        if len(messages) == 0:
            continue

        quality_scores = [m['quality'] for m in messages if not np.isnan(m.get('quality', np.nan))]
        ranks = [m['rank'] for m in messages if not np.isnan(m.get('rank', np.nan))]

        trees.append({
            'tree_id': root_id,
            'messages': messages,
            'texts': [m['text'] for m in messages],
            'num_turns': len(messages),
            'roles': [m['role'] for m in messages],
            'avg_quality': np.mean(quality_scores) if quality_scores else np.nan,
            'avg_rank': np.mean(ranks) if ranks else np.nan,
            'has_quality_labels': len(quality_scores) > 0,
        })

    trees_df = pd.DataFrame(trees)
    rated = trees_df['has_quality_labels'].sum()
    print(f"  Built {len(trees_df):,} conversation trees "
          f"({rated:,} with quality ratings)")
    return trees_df


def get_quality_labeled_subset(trees_df: pd.DataFrame) -> pd.DataFrame:
    """Return only trees that have human quality annotations."""
    labeled = trees_df[trees_df['has_quality_labels']].copy()
    print(f"  Quality-labeled subset: {len(labeled):,} trees "
          f"(avg quality: {labeled['avg_quality'].mean():.3f})")
    return labeled
