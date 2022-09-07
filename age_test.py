import torch
from face_dataset import FaceDataset
from torch.utils.data import DataLoader


def test(device):
    batch_size = 100
    cid = FaceDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/age.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    loss_cum = 0
    itr = 0
    results = []

    for (image, age, gender) in dataloader:
        image = image.to(device)
        age = age.to(device)
        y_hat = model(image)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, age)
        itr = itr+1
        loss_cum = loss_cum + loss.item()

        for i in range(y_hat.shape[0]):
            results.append((age[i].item(), y_hat[i].item()))

    gt2 = [i[0] for i in results]
    hat2 = [i[1] for i in results]
    gt = cid.unscale(gt2)
    hat= cid.unscale(hat2)
    print(f"Actual Age\t\t\tPredicted Age")
    for i in range(len(gt)):
        actual = f"{gt[i]:.1f}".ljust(20)
        predicted = f"{hat[i]:.1f}".ljust(20)
        print(f"{actual}{predicted}")

    loss_cum = loss_cum / itr
    print(f"Loss {loss_cum:.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
