"""
Data Loader and Preprocessing Module for Breast Cancer Wisconsin (Diagnostic) Dataset.
Provides functions to load raw data, perform stratified train-test splitting (to prevent leakage),
and return feature names and class distributions.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset(as_dataframe: bool = True):
    """
    Loads the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    Features: 30 numeric continuous cellular morphology measurements.
    Target: 1 = Malignant, 0 = Benign (Mapped for standard clinical risk modeling).
    
    Note: In standard sklearn, target 0 = Malignant, 1 = Benign.
    We invert this mapping so that 1 = Malignant (the positive risk class)
    and 0 = Benign (the negative class) to align with standard clinical diagnostic conventions.
    """
    raw_data = load_breast_cancer(as_frame=as_dataframe)
    X = raw_data.data.copy()
    
    # Invert target so 1 = Malignant, 0 = Benign
    y = pd.Series(1 - raw_data.target, name="diagnosis")
    
    target_names = {1: "Malignant", 0: "Benign"}
    return X, y, target_names


def get_stratified_split(X, y, test_size: float = 0.20, random_state: int = 42):
    """
    Splits the dataset into a train set and a strictly held-out test set.
    Preserves class balance via stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y, target_names = load_dataset()
    print(f"Dataset Loaded Successfully!")
    print(f"Total Samples: {X.shape[0]}, Total Features: {X.shape[1]}")
    print(f"Class Distribution: Malignant={sum(y==1)} ({sum(y==1)/len(y):.1%}), Benign={sum(y==0)} ({sum(y==0)/len(y):.1%})")
