import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def train_image_classifier(model, dataset, learning_rate, batch_size, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, labels = batch['X'], batch['y']
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')