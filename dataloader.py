import os
from typing import Iterable, Optional

import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from augmentation import augmentation_pipeline

def one_hot(item: int, n: int):
    ret = np.zeros(n)
    ret[item] = 1
    return ret

class MRIDataLoader(tf.keras.utils.Sequence):

    def __init__(self,
                 data_path: str,
                 metadata_path: str,
                 patients: Optional[Iterable[str]] = None,
                 batch_size: int = 32,
                 shuffle_all: bool = True,
                 shuffle_batch: bool = True,
                 verbose: bool = True,
                 augment_data: bool = False):
        super(MRIDataLoader, self).__init__()

        self.data_path = data_path
        self.metadata_path = metadata_path
        self.verbose = verbose
        self.augment_data = augment_data

        self.metadata = pd.read_csv(metadata_path)

        if patients is None:
            self.patients = sorted(os.listdir(data_path))
        else:
            self.patients = list(patients)

        if shuffle_all:
            np.random.shuffle(self.patients)

        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.shuffle_batch = shuffle_batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.patients) / self.batch_size))
    
    def preprocess_data_batch(self, data_batch: Iterable):
        data_batch = tf.cast(np.stack(data_batch), dtype=tf.float32)
        if self.augment_data:
            data_batch = augmentation_pipeline(images=data_batch)
        return data_batch

    def preprocess_label_batch(self, label_batch: Iterable):
        label_batch = tf.cast(np.stack(label_batch), dtype=tf.float32)
        return label_batch

    def __getitem__(self, item: int):
        if self.batch_size * (item + 1) >= len(self.patients):
            batch_patients = self.patients[self.batch_size * item:]
        else:
            batch_patients = self.patients[self.batch_size * item:self.batch_size * (item + 1)]

        if self.shuffle_batch:
            np.random.shuffle(batch_patients)

        data_batch = []
        label_batch = []
        for patient in batch_patients:
            if self.verbose:
                print(f"Loading patient {patient}.")
            patient_dir = os.path.join(self.data_path, patient)
            try:
                data_batch.append(imageio.imread(patient_dir, pilmode="RGB"))                
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"Missing file for patient {patient}: {e.filename}")

            label_batch.append(one_hot(self.metadata[self.metadata["name"] == patient]["label"].iloc[0], n=len(self.metadata["label"].unique())))

        data_batch = self.preprocess_data_batch(data_batch)
        label_batch = self.preprocess_label_batch(label_batch)

        return data_batch, label_batch
    
    