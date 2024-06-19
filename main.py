import argparse
import os
from pathlib import Path
import json
import pickle as pkl
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import convert_labels_to_tensors, label_to_ind
from preprocessing import text_files_to_dataframe, date_to_interval, skip_date_intervals, vectorize_corpus
from models import MonoTaskNNS, MonoTaskNNSND
from train_test_models import train_sex_classification, test_sex_classification, train_sex_date_classification, test_sex_date_classification

def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    model_default_params_path = str(Path(project_folder, "defaul_model_params.json"))

    ap.add_argument("-mu", "--model-usage", default="train", help="Specify usage of given model", type=str, choices=["train", "test"])
    ap.add_argument("-m", "--model", default="sexe", help="Specify which model you want to use", type=str, choices=["sexe", "multi-head"])
    ap.add_argument("-dloss", "--new-date-loss", default=False, help="Specify usage of new date loss", type=bool)
    ap.add_argument("-wdloss", "--date-loss-weight", default=0.2, help="Specify date loss weight", type=float)
    ap.add_argument("-mp", "--model-params", default=model_default_params_path, help="Specify model parameters", type=str)

    args = ap.parse_args()

    project_folder = str(Path(os.getcwd()))
    data_df_folder = str(Path(project_folder, "data"))

    if not os.path.isdir(data_df_folder):
        os.makedirs(data_df_folder)

    if not os.path.isdir(str(Path(project_folder, "models"))):
        os.makedirs(str(Path(project_folder, "models")))

    raw_data = text_files_to_dataframe()
    raw_data = date_to_interval(raw_data)
    raw_data = skip_date_intervals(1825, raw_data)

    raw_data["text_encoded"] = vectorize_corpus(raw_data)

    text_data = raw_data[["text_encoded", "sexe", "date", "date_class", "file_name"]]
    slabels_ind_map, dlabels_ind_map = label_to_ind(text_data["date_class"])
    text_data["sexe"] = text_data["sexe"].map(lambda x: slabels_ind_map[x])
    text_data["date_class"] = text_data["date_class"].map(lambda x: dlabels_ind_map[x])

    # Save RAM memory
    del raw_data

    val_ratio = 0.15
    test_ratio = 0.10
    train_df, test_df = train_test_split(raw_data, test_size=test_ratio, random_state=42, stratify=text_data["labels"])
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=42, stratify=train_df["sexe"])

    del text_data

    with open(data_df_folder+"train_df.pkl", "wb") as fout:
        pkl.dump(train_df, fout, protocol=pkl.HIGHEST_PROTOCOL)
    with open(data_df_folder+"val_df.pkl", "wb") as fout:
        pkl.dump(val_df, fout, protocol=pkl.HIGHEST_PROTOCOL)
    with open(data_df_folder+"test_df.pkl", "wb") as fout:
        pkl.dump(test_df, fout, protocol=pkl.HIGHEST_PROTOCOL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.model_params) as params_loader:
            model_default_params = json.load(params_loader)
        
    batch_size = model_default_params["batch_size"]
    lr = model_default_params["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_df = pd.read_pickle(str(Path(data_df_folder, "train_df.pkl")))
    val_df = pd.read_pickle(str(Path(data_df_folder, "val_df.pkl")))
    test_df = pd.read_pickle(str(Path(data_df_folder, "test_df.pkl")))

    train_sex_labels_tensor, train_date_labels_tensor, train_date_tensor = convert_labels_to_tensors(train_df)
    val_sex_labels_tensor, val_date_labels_tensor, val_date_tensor = convert_labels_to_tensors(val_df)
    test_sex_labels_tensor, test_date_labels_tensor, test_date_tensor = convert_labels_to_tensors(test_df)

    train_data = np.array(train_df['text_encoded'].to_list(), dtype=np.float64)
    val_data = np.array(val_df['text_encoded'].to_list(), dtype=np.float64)
    test_data = np.array(test_data['text_encoded'].to_list(), dtype=np.float64)

    train_data_tensor = torch.FloatTensor(train_data)
    val_data_tensor = torch.FloatTensor(val_data)
    test_data_tensor = torch.FloatTensor(test_data)

    if args.model_usage == "train":

        if args.model == "sexe":
            train_dataset = TensorDataset(train_data_tensor, train_sex_labels_tensor)
            val_dataset = TensorDataset(val_data_tensor, val_sex_labels_tensor)
            test_dataset = TensorDataset(test_data_tensor, test_sex_labels_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = MonoTaskNNS(train_loader.shape[0], train_sex_labels_tensor.unique())
            model.to(device)

            best_epoch, train_metrics, val_metrics = train_sex_classification(train_loader, val_loader, model, device, _, _)

            acc, pre, rec, f1 = test_sex_classification(test_loader, model, device)

            user_ans = ""

            while (user_ans != "y") or (user_ans != "n"):
                user_ans = input("Do you to save your model ? (y/n)").lower()

                if user_ans == "y":
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, str(Path(project_folder, "models", f"s_model_MLP_DI_{best_epoch}.pth")))
                
                elif user_ans == "n":
                    break
                else:
                    print("Give a proper answre please (y/n).")
        
        if args.model == "multi-head":
            train_dataset = TensorDataset(train_data_tensor, train_sex_labels_tensor, train_date_labels_tensor, train_date_tensor)
            val_dataset = TensorDataset(val_data_tensor, val_sex_labels_tensor, val_date_labels_tensor, val_date_tensor)
            test_dataset = TensorDataset(test_data_tensor, test_sex_labels_tensor, test_date_labels_tensor, test_date_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = MonoTaskNNSND(train_loader.shape[0], train_sex_labels_tensor.unique(), train_date_labels_tensor.unique())
            model.to(device)

            criterion_sex = nn.CrossEntropyLoss()
            criterion_date = nn.CrossEntropyLoss()

            best_epoch, train_sex_metrics, val_sex_metrics, train_date_metrics, val_date_metrics = train_sex_date_classification(train_loader, val_loader, model, device, criterion_sex, criterion_date, args.new_date_loss)

            acc_sex, pre_sex, rec_sex, f1_sex, acc_date, mae_date = test_sex_date_classification(test_loader, model, device)

            user_ans = ""

            while (user_ans != "y") or (user_ans != "n"):
                user_ans = input("Do you to save your model ? (y/n)").lower()

                if user_ans == "y":
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, str(Path(project_folder, "models", f"s_model_MLP_DI_{best_epoch}.pth")))
                
                elif user_ans == "n":
                    break
                else:
                    print("Give a proper answre please (y/n).")
    
    if args.model_usage == "test":
        if args.model == "sexe":
            test_dataset = TensorDataset(test_data_tensor, test_sex_labels_tensor)

            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = MonoTaskNNS(train_loader.shape[0], train_sex_labels_tensor.unique())
            model.to(device)

            model.load_state_dict(torch.load(str(Path(project_folder, "models", "best_sex_classifier.pth")))["model_state_dict"])

            test_sex_classification(test_loader, model, device)
        
        if args.model == "multi-head":
            test_dataset = TensorDataset(test_data_tensor, test_sex_labels_tensor, test_date_labels_tensor, test_date_tensor)

            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = MonoTaskNNSND(train_loader.shape[0], train_sex_labels_tensor.unique(), train_date_labels_tensor.unique())
            model.to(device)

            if args.new_date_loss:
                model.load_state_dict(torch.load(str(Path(project_folder, "models", "best_multi_head_new_loss.pth")))["model_state_dict"])
            else:
                model.load_state_dict(torch.load(str(Path(project_folder, "models", "best_multi_head.pth")))["model_state_dict"])

            test_sex_date_classification(test_loader, model, device)

if __name__ == "__main__":
    main()
