import torch

import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score

from models import EarlyStopping

def train_sex_classification(train_loader, val_loader, model, device, optimizer, criterion_sex):
    # Training loop
    num_epochs = 50

    sex_early_stopping = EarlyStopping(tolerance=3)
    pred_sex_val_loss = 0.0

    train_sex_losses = []
    train_sex_acc = []
    val_sex_losses = []
    val_sex_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    best_epoch = 0

    for epoch in range(num_epochs):

        train_correct_sex = 0
        train_total = 0

        model.train()
        train_sex_loss = 0.0
        for inputs, sex_labels in train_loader:
            inputs, sex_labels = inputs.to(device), sex_labels.to(device)

            optimizer.zero_grad()

            outputs_sex = model(inputs)
            sex_loss = criterion_sex(outputs_sex, sex_labels)
            sex_loss.backward()
            optimizer.step()

            train_sex_loss += sex_loss.item()

            _, predicted_sex = torch.max(outputs_sex, 1)

            train_total += sex_labels.size(0)
            train_correct_sex += (predicted_sex == sex_labels).sum().item()

        print(f"Epoch {epoch}/{num_epochs}")
        print("TRAINING")
        train_average_sex_loss = train_sex_loss / len_train_loader
        train_sex_losses.append(train_average_sex_loss)
        print(f"Sex Loss: {train_average_sex_loss}")

        train_accuracy_sex = train_correct_sex / train_total
        train_sex_acc.append(train_accuracy_sex)
        print(f"Train Accuracy (Sex): {train_accuracy_sex}")

        val_correct_sex = 0
        val_total = 0

        ###########################
        # Validation

        model.eval()

        with torch.no_grad():
            val_sex_loss = 0.0
            for inputs, sex_labels in val_loader:
                inputs, sex_labels = inputs.to(device), sex_labels.to(device)

                outputs_sex = model(inputs)

                sex_loss = criterion_sex(outputs_sex, sex_labels)

                val_sex_loss += sex_loss.item()

                _, predicted_sex = torch.max(outputs_sex, 1)

                val_total += sex_labels.size(0)
                val_correct_sex += (predicted_sex == sex_labels).sum().item()

        print("VALIDATION")
        val_average_sex_loss = val_sex_loss / len_val_loader
        val_sex_losses.append(val_average_sex_loss)
        print(f"Sex Loss: {val_average_sex_loss}")

        val_accuracy_sex = val_correct_sex / val_total
        val_sex_acc.append(val_accuracy_sex)
        print(f"Validation Accuracy (Sex): {val_accuracy_sex}")

        ###########################
        # Early stopping
        sex_early_stopping(pred_sex_val_loss, val_average_sex_loss)
        if sex_early_stopping.early_stop:
            print(f"We are at epoch {epoch} from sex loss")
            best_epoch = epoch
            break

        pred_sex_val_loss = val_average_sex_loss

    return best_epoch, (train_sex_acc, train_sex_losses), (val_sex_acc, val_sex_losses)

def test_sex_classification(test_loader, model, device):
    y_pred = []
    y_true = []

    correct_sex = 0
    total = 0

    model.eval()
    with torch.no_grad():
            for inputs, sex_labels in test_loader:
                inputs, sex_labels = inputs.to(device), sex_labels.to(device)

                outputs_sex = model(inputs)
                _, predicted_sex = torch.max(outputs_sex, 1)
                y_pred.extend(predicted_sex.data.cpu().numpy())
                y_true.extend(sex_labels.data.cpu().numpy())

                total += sex_labels.size(0)
                correct_sex += (predicted_sex == sex_labels).sum().item()

    accuracy_sex = correct_sex / total
    print(f"Test Accuracy (Sex): {accuracy_sex}")
    s_precision = precision_score(y_true, y_pred)
    s_recall = recall_score(y_true, y_pred)
    s_f1_score = f1_score(y_true, y_pred)
    print(f"Sex Precision: {s_precision}")
    print(f"Sex Recall: {s_recall}")
    print(f"Sex F1 Score: {s_f1_score}")

    return accuracy_sex, s_precision, s_recall, s_f1_score

def train_sex_date_classification(train_loader, val_loader, model, device, optimizer, criterion_sex, criterion_date, new_loss=False):
    train_sex_losses = []
    train_date_losses = []
    val_sex_losses = []
    val_date_losses = []
    train_sex_acc = []
    train_date_acc = []
    val_sex_acc = []
    val_date_acc = []
    if new_loss:
        train_date_diff = []
        val_date_diff = []

    sex_early_stopping = EarlyStopping(tolerance=3)
    pred_sex_val_loss = 0.0

    # Training loop
    num_epochs = 50
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    best_epoch = 0

    for epoch in range(num_epochs):

        train_correct_sex = 0
        train_correct_date = 0
        train_mae_date = []
        train_total = 0

        model.train()
        train_sex_loss = 0.0
        train_date_loss = 0.0
        for inputs, sex_labels, date_interval_labels, date_labels in train_loader:
            inputs, sex_labels, date_interval_labels, date_labels = inputs.to(device), sex_labels.to(device), date_interval_labels.to(device), date_labels.to(device)

            optimizer.zero_grad()

            outputs_sex, outputs_date = model(inputs)
            sex_loss = criterion_sex(outputs_sex, sex_labels)

            if new_loss:
                unique_label_interval = torch.unique(date_interval_labels).tolist()
                unique_date_interval = torch.tensor([1825 + (i*25) for i in unique_label_interval])
                weighted_date_preds = date_preds * unique_date_interval
                new_date_preds = torch.sum(weighted_date_preds, dim=1)
                train_mse_date_loss = torch.mean(torch.pow(new_date_preds - date_labels, 2))
                date_loss = 0.00003*train_mse_date_loss

                mae_date = torch.abs(new_date_preds - date_labels)
                train_mae_date.extend(mae_date.tolist())
            else:
                date_loss = 0.2*criterion_date(outputs_date, date_interval_labels)
            loss = sex_loss + date_loss
            loss.backward()
            optimizer.step()

            train_sex_loss += sex_loss.item()
            train_date_loss += date_loss.item()

            _, predicted_sex = torch.max(outputs_sex, 1)
            _, predicted_date = torch.max(outputs_date, 1)

            train_total += sex_labels.size(0)
            train_correct_sex += (predicted_sex == sex_labels).sum().item()
            train_correct_date += (predicted_date == date_interval_labels).sum().item()

        print(f"Epoch {epoch}/{num_epochs}")
        print("TRAINING")
        train_average_sex_loss = train_sex_loss / len_train_loader
        train_sex_losses.append(train_average_sex_loss)
        print(f"Sex Loss: {train_average_sex_loss}")
        train_average_date_loss = train_date_loss / len_train_loader
        train_date_losses.append(train_average_date_loss)
        print(f"Date Loss: {train_average_date_loss}")

        train_accuracy_sex = train_correct_sex / train_total
        train_sex_acc.append(train_accuracy_sex)
        train_accuracy_date = train_correct_date / train_total
        train_date_acc.append(train_accuracy_date)
        print(f"Train Accuracy (Sex): {train_accuracy_sex}")
        print(f"Train Accuracy (Date): {train_accuracy_date}")

        if new_loss:
            train_average_date_mae = np.mean(train_mae_date)
            train_date_diff.append(train_average_date_mae)
            print(f"Train Difference (Date): {train_average_date_mae}")

        val_correct_sex = 0
        val_correct_date = 0
        val_mae_date = []
        val_total = 0

        ###########################
        # Validation

        model.eval()

        with torch.no_grad():
            val_sex_loss = 0.0
            val_date_loss = 0.0
            for inputs, sex_labels, date_interval_labels, date_labels in val_loader:
                inputs, sex_labels, date_interval_labels, date_labels = inputs.to(device), sex_labels.to(device), date_interval_labels.to(device), date_labels.to(device)

                outputs_sex, outputs_date = model(inputs)
                sex_loss = criterion_sex(outputs_sex, sex_labels)
                date_preds = torch.softmax(outputs_date, dim=1)
                if new_loss:
                    unique_label_interval = torch.unique(date_interval_labels).tolist()
                    unique_date_interval = torch.tensor([1825 + (i*25) for i in unique_label_interval])
                    weighted_date_preds = date_preds * unique_date_interval
                    new_date_preds = torch.sum(weighted_date_preds, dim=1)
                    val_mse_date_loss = torch.mean(torch.pow(new_date_preds - date_labels, 2))
                    date_loss = 0.00003*val_mse_date_loss

                    mae_date = torch.abs(new_date_preds - date_labels)
                    val_mae_date.extend(mae_date.tolist())
                else:
                    date_loss = 0.2*criterion_date(outputs_date, date_interval_labels)
                

                val_sex_loss += sex_loss.item()
                val_date_loss += date_loss.item()

                _, predicted_sex = torch.max(outputs_sex, 1)
                _, predicted_date = torch.max(outputs_date, 1)

                val_total += sex_labels.size(0)
                val_correct_sex += (predicted_sex == sex_labels).sum().item()
                val_correct_date += (predicted_date == date_interval_labels).sum().item()

            print("VALIDATION")
            val_average_sex_loss = val_sex_loss / len_val_loader
            val_sex_losses.append(val_average_sex_loss)
            print(f"Sex Loss: {val_average_sex_loss}")
            val_average_date_loss = val_date_loss / len_val_loader
            val_date_losses.append(val_average_date_loss)
            print(f"Date Loss: {val_average_date_loss}")

            val_accuracy_sex = val_correct_sex / val_total
            val_sex_acc.append(val_accuracy_sex)
            val_accuracy_date = val_correct_date / val_total
            val_date_acc.append(val_accuracy_date)
            print(f"Validation Accuracy (Sex): {val_accuracy_sex}")
            print(f"Validation Accuracy (Date): {val_accuracy_date}")

            if new_loss:
                val_average_date_mae = np.mean(val_mae_date)
                val_date_diff.append(val_average_date_mae)
                print(f"Validation Difference (Date): {val_average_date_mae}")

            ###########################
            # Early stopping
            sex_early_stopping(pred_sex_val_loss, val_average_sex_loss)
            if sex_early_stopping.early_stop:
                print(f"We are at epoch {epoch+1} from sex loss")
                best_epoch = epoch
                break

        pred_sex_val_loss = val_average_sex_loss

    return best_epoch, (train_sex_acc, train_sex_losses), (val_sex_acc, val_sex_losses), (train_date_acc, train_date_losses), (val_date_acc, val_date_losses)


def test_sex_date_classification(test_loader, model, device, new_loss=False):
    y_pred = []
    y_true = []

    correct_sex = 0
    correct_date = 0
    test_maes_date = []
    total = 0

    model.eval()
    with torch.no_grad():
            for inputs, sex_labels, date_interval_labels, date_labels in test_loader:
                inputs, sex_labels, date_interval_labels, date_labels = inputs.to(device), sex_labels.to(device), date_interval_labels.to(device), date_labels.to(device)

                outputs_sex, outputs_date = model(inputs)
                date_preds = torch.softmax(outputs_date, dim=1)
                if new_loss:
                    unique_label_interval = torch.unique(date_interval_labels).tolist()
                    unique_date_interval = torch.tensor([1825 + (i*25) for i in unique_label_interval])
                    weighted_date_preds = date_preds * unique_date_interval
                    new_date_preds = torch.sum(weighted_date_preds, dim=1)

                _, predicted_sex = torch.max(outputs_sex, 1)
                y_pred.extend(predicted_sex.data.cpu().numpy())
                y_true.extend(sex_labels.data.cpu().numpy())
                _, predicted_date = torch.max(outputs_date, 1)

                total += sex_labels.size(0)
                correct_sex += (predicted_sex == sex_labels).sum().item()
                correct_date += (predicted_date == date_interval_labels).sum().item()
                if new_loss:
                    test_maes_date.extend(torch.abs(new_date_preds - date_labels).tolist())
                else:
                    test_maes_date.extend(torch.abs(date_preds - date_labels).tolist())

    accuracy_sex = correct_sex / total
    accuracy_date = correct_date / total
    diff_date = np.mean(test_maes_date)
    print(f"Test Accuracy (Sex): {accuracy_sex}")
    print(f"Test Accuracy (Date): {accuracy_date}")
    print(f"Test Difference (Date): {diff_date}")

    s_precision = precision_score(y_true, y_pred)
    s_recall = recall_score(y_true, y_pred)
    s_f1_score = f1_score(y_true, y_pred)
    print(f"Sex Precision: {s_precision}")
    print(f"Sex Recall: {s_recall}")
    print(f"Sex F1 Score: {s_f1_score}")

    return accuracy_sex, s_precision, s_recall, s_f1_score, accuracy_date, diff_date
