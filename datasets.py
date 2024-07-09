import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_digits_dataset():
    data = load_digits()
    X, y = data.images, data.target
    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

class FlattenedImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return dict(X=self.X[idx], y=self.y[idx])

    def __len__(self):
        return self.X.shape[0]