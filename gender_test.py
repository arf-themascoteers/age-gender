import torch
from face_dataset import FaceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def test(device):
    batch_size = 100
    cid = FaceDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/gender.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    loss_cum = 0
    itr = 0
    results = []

    for (image, age, gender) in dataloader:
        image = image.to(device)
        gender = gender.to(device)
        y_hat = model(image)
        loss = F.cross_entropy(y_hat, gender)
        itr = itr+1
        loss_cum = loss_cum + loss.item()
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(gender.data.view_as(pred)).sum()
        total += gender.shape[0]

    print(f'Total:{total}, Correct:{correct}, Accuracy:{correct / total * 100:.2f}')
    loss_cum = loss_cum / itr
    print(f"Loss {loss_cum:.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
