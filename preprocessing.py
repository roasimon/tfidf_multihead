import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import merge_to_numpy

def text_files_to_dataframe() -> pd.DataFrame:
    files_folder = str(Path(os.getcwd(), "corpus_multihead"))
    text_files_list = os.listdir(files_folder)
    raw_data = pd.DataFrame(columns=["text", "sexe", "date"])

    for file in text_files_list:
        path = f"{files_folder}{file}"
        if not os.path.isfile(path):
            continue
        # read file
        with open(path, 'r', encoding='utf-8') as f:
            file_elements = file.split("(")
            sexe = file_elements[4].split(")")[0]
            date = file_elements[5].split(")")[0]
            text = f.readlines()[0]
            nb_carac_text = len(text)

            tmp_df = pd.DataFrame({"file_name": [file], "text": [text], "len_text": [nb_carac_text], "sexe": ["homme" if sexe == "1" else "femme"], "date": [int(date)]})
            raw_data = pd.concat([raw_data, tmp_df], ignore_index=True)
    
    return raw_data

def date_to_interval(raw_data: pd.DataFrame) -> pd.DataFrame:
    current_interval = 1500
    while current_interval != 2050:
        raw_data.loc[(current_interval <= raw_data["date"]) & (raw_data["date"] < current_interval+25), "date_class"] = current_interval
        current_interval += 25
    
    raw_data["labels"] = raw_data["sexe"].astype(str) +"_"+ raw_data["date_class"].astype(str)
    
    return raw_data

def skip_date_intervals(min_date, raw_data):
    raw_data[["len_text", "date_class"]] = raw_data[["len_text", "date_class"]].applymap(np.int64)
    raw_data = raw_data.loc[raw_data["date_class"] >= min_date].reset_index(drop=True)


def vectorize_corpus(data_df):
    input_text = data_df["text"].to_numpy()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(input_text) 
    df = pd.DataFrame(X.todense())
    return df.apply(merge_to_numpy, axis=1)