#creates folders and subfolders for the dataset (not necessary but certainly fun)
import os
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Space", "Delete"]
base_dir = "dataset"

for label in labels:
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)