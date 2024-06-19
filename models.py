import torch.nn as nn

class MonoTaskNNS(nn.Module):
  def __init__(self, input_size, num_sex_classes):
    super(MonoTaskNNS, self).__init__()
    self.fc1_author_sex = nn.Linear(input_size, 1024)

    self.fc2_author_sex = nn.Linear(1024, 512)

    self.fc3_author_sex = nn.Linear(512, 128)

    self.fc4_author_sex = nn.Linear(128, num_sex_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.fc1_author_sex(x))

      x = self.relu(self.fc2_author_sex(x))

      x = self.relu(self.fc3_author_sex(x))

      author_sex_output = self.fc4_author_sex(x)
      return author_sex_output
  
class MonoTaskNNSND(nn.Module):
  def __init__(self, input_size, num_sex_classes, num_date_classes):
    super(MonoTaskNNSND, self).__init__()
    self.fc1_author_sex = nn.Linear(input_size, 1024)

    self.fc2_author_sex = nn.Linear(1024, 512)

    self.fc3_author_sex = nn.Linear(512, 128)

    self.fc4_author_sex = nn.Linear(128, num_sex_classes)
    self.fc4_publication_year = nn.Linear(128, num_date_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.fc1_author_sex(x))

      x = self.relu(self.fc2_author_sex(x))

      x = self.relu(self.fc3_author_sex(x))

      author_sex_output = self.fc4_author_sex(x)
      publication_year_output = self.fc4_publication_year(x)
      return author_sex_output, publication_year_output
  
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0.5):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, pred_loss, current_loss):
        if abs(current_loss - pred_loss) < self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True
