from typing import Tuple,Dict
from Metadata import SplitMetadata
import random

def split(records: Dict[str, Dict], split_metadata: SplitMetadata, seed: int = 42):
    random.seed(seed)

    episodes = list(records.keys())
    patients = sorted({episode.split("_")[0] for episode in episodes})

    n_total = len(patients)
    n_train = int(n_total * split_metadata.train)
    n_test = int(n_total * split_metadata.test)

    train_patients = random.sample(patients, n_train)

    remaining = [p for p in patients if p not in train_patients]
    test_patients = random.sample(remaining, n_test)

    val_patients = [p for p in remaining if p not in test_patients]

    train_records = {k: v for k, v in records.items() if k.split("_")[0] in train_patients}
    test_records = {k: v for k, v in records.items() if k.split("_")[0] in test_patients}
    val_records = {k: v for k, v in records.items() if k.split("_")[0] in val_patients}

    return train_records, test_records, val_records