from argparse import Namespace
import json
import os
from typing import Iterable, Union

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from dataloader import MRIDataLoader
from models.models import MODELS


def build_model(model_name: str,
                optimizer: Union[str, tf.keras.optimizers.Optimizer],
                loss: Union[str, tf.keras.losses.Loss],
                metrics: Iterable[tf.keras.metrics.Metric],
                model_params=None) -> tf.keras.Model:

    if model_params is None:
        model_params = {}

    model_class = MODELS[model_name]
    
    model = model_class(**model_params)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, 128, 128, 3))

    print(model.summary())
    return model


def train(namespace: Namespace) -> None:
        
    results_dir = os.path.join(namespace.results_path, namespace.training_id)
    os.makedirs(results_dir, exist_ok=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                      patience=namespace.early_stopping_patience, 
                                                      restore_best_weights=True, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(results_dir, "log.csv"))

    callbacks = [early_stopping, csv_logger]
    
    model = build_model(namespace.model_name,
                        optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["accuracy", tf.keras.metrics.AUC()])
    
    print(f"Start training of model {namespace.training_id}.")
    
    metadata = pd.read_csv(os.path.join(namespace.data_path, "metadata.csv"))
    X_train, X_test, y_train, y_test = train_test_split(metadata["name"], metadata["label"], test_size=0.3,
                                            stratify=metadata["label"], random_state=namespace.random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.33,
                                            stratify=y_test, random_state=namespace.random_seed)
    
    loader = MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"), metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                           patients=X_train, batch_size=namespace.batch_size, verbose=False, augment_data=False)
    validation_loader = MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"), metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                                      patients=X_val, shuffle_all=False, shuffle_batch=False, batch_size=namespace.batch_size, verbose=False, augment_data=False)
    test_loader = MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"), metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                                      patients=X_test, shuffle_all=False, shuffle_batch=False, batch_size=namespace.batch_size, verbose=False, augment_data=False)

    history = model.fit(loader, validation_data=validation_loader, epochs=namespace.epochs, callbacks=callbacks)

    print(f"Training of model {namespace.training_id} finished.")

    model.save(os.path.join(results_dir, "saved_model"))
    with open(os.path.join(results_dir, "history.json"), "w") as file:
        json.dump(history.history, file, indent=4)
    with open(os.path.join(results_dir, "args.json"), "w") as file:
        json.dump(namespace.__dict__, file, indent=4)
        
    train_predictions = model.predict(MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"),
                                                    metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                                                    patients=X_train, shuffle_all=False, shuffle_batch=False, batch_size=256,
                                                    verbose=False, augment_data=False)).argmax(axis=1)
    pd.DataFrame({"name": X_train, "prediction": train_predictions}).to_csv(os.path.join(results_dir, "train_predictions.csv"), index=False)    
    
    validation_predictions = model.predict(validation_loader).argmax(axis=1)
    pd.DataFrame({"name": X_val, "prediction": validation_predictions}).to_csv(os.path.join(results_dir, "validation_predictions.csv"), index=False)    

    test_predictions = model.predict(test_loader).argmax(axis=1)
    pd.DataFrame({"name": X_test, "prediction": test_predictions}).to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)    