import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_image_classifier(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, predicted.numpy()))