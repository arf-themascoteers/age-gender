import torch
import torch.nn.functional as F
from face_dataset import FaceDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from multi_machine import MultiMachine

def train(device):
    batch_size = 500
    cid = FaceDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = MultiMachine()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')
    num_epochs = 10
    n_batches = int(len(cid)/batch_size) + 1
    batch_number = 0
    loss = None

    for epoch in range(num_epochs):
        batch_number = 0
        for (image, age, gender) in dataloader:
            image = image.to(device)
            age = age.to(device)
            gender = gender.to(device)
            optimizer.zero_grad()
            y_hat = model(image)
            age_hat = y_hat[:, 0]
            gender_hat = y_hat[:, 1:]
            age_hat = age_hat.reshape(-1)
            age_loss = criterion(age_hat, age)
            gender_loss = F.cross_entropy(gender_hat, gender)
            loss = age_loss + gender_loss
            loss.backward()
            optimizer.step()
            batch_number += 1
            print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.4f}')

    print("Train done")
    torch.save(model, 'models/multi.h5')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)