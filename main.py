import torch
import age_train
import age_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training started...")
age_train.train(device)

print("Testing started...")
age_test.test(device)