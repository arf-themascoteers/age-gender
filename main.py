import torch
import multi_train
import multi_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training started...")
multi_train.train(device)

print("Testing started...")
multi_test.test(device)