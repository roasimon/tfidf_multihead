import os
from pathlib import Path
import shutil
import numpy as np
import torch

def create_set_folder(data_df, current_folder_path, dest_folder):
  for file_name in data_df["file_name"]:
    current_file_path = str(Path(current_folder_path, file_name))
    new_file_path = str(Path(current_folder_path, dest_folder, file_name))
    shutil.copyfile(current_file_path, new_file_path)

def merge_to_numpy(row):
  return row.to_numpy(dtype=np.float64)

def label_to_ind(date_column):
  sex_labels_index = {
      2: 0,
      1: 1
  }

  unique_date_values = sorted(date_column.unique())

  date_labels_index = {k: v for v, k in enumerate(unique_date_values)}

  return sex_labels_index, date_labels_index

def convert_labels_to_tensors(df):
    sex_labels_tensor = torch.LongTensor(df["sexe"].values)
    date_labels_tensor = torch.LongTensor(df["date_class"].values)
    date_tensor = torch.LongTensor(df["date"].to_numpy(dtype=np.int64))

    return sex_labels_tensor, date_labels_tensor, date_tensor
