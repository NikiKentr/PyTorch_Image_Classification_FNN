import torch
from datasets import load_digits_dataset, FlattenedImageDataset
from models import FNN
from training import train_image_classifier
from evaluation import evaluate_image_classifier

# Load digits dataset and initialize FNN model
X_train, X_test, y_train, y_test = load_digits_dataset()
dataset_train = FlattenedImageDataset(X_train, y_train)
input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(np.unique(y_train))
fnn = FNN(input_size, hidden_size, num_classes)

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 100

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and evaluate FNN model
train_image_classifier(fnn, dataset_train, learning_rate, batch_size, epochs, device)
evaluate_image_classifier(fnn, X_test, y_test, device)