#torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

#models
import torchvision.models.segmentation as models

#utils 
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#local 
from utils import dataset, random_points
from loss import partial_cross_entropy

## define args
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)  
args = parser.parse_args()
print(args)

## set seeds + device
torch.manual_seed(522)
np.random.seed(522)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load data
## create dataset
## create dataloader
## TODO: load splitted data
# actually using the validation split for initial shape tests
train_dataset = dataset.LoveDADataset(root_dir=args.root, split="train", transform=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# valid_dataset
# valid_loader
# test_dataset
# test_loader


## define model (TODO: later will implement a CNN model)
model = models.deeplabv3_resnet50(pretrained=False)  
model.eval() 
#swap last layer for correct number of classes (7)
model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1)) #7 = num_class
# print(model.classifier)
model = model.to(device)

## define loss, optimizer
#criterion = nn.CrossEntropyLoss()
criterion = partial_cross_entropy.PartialCrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

## TODO: define variable for metrics (right now will use only loss)
best_valid_loss = 0

## loop data (epochs)
for epoch in tqdm(range(args.epoch)):
    model.train()
    for i, (img, msk) in enumerate(tqdm(train_loader)):
        img, msk = img.to(device), msk.to(device)
        # print(img.size()) # (batch_size, channels, h, w) (10, 3, 129, 128)
        # print(msk.size())

        ## random samples for the mask 

        # am i actually putting a lot of 0 points, but i guess it needs to be -1?
        #because 0 is a class, no?
        random_mask = random_points.create_random_points(msk, 0.3, 1) 
        random_mask = random_mask.squeeze(1).long().to(device=device)
        # random_points.plot_mask(random_mask, 0)
        # random_points.plot_mask(random_mask, 1)

        # print(random_mask)

        output = model(img)['out']
        # print(output.size()) # (batch_size, N_CLASSES, H, W)
        # print(random_mask.size())
        # print("hi")

        loss = criterion(output, random_mask)  
        loss.backward()
        optimizer.step()

        print(f"[TRAIN] epoch {epoch + 1}/{args.epoch} batch loss: {loss.item():.4f}")

    ## TODO: validation loss
    valid_loss = 0

    ## save checkpoint
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "checkpoints/test.pth")

## TODO: test model (load from best checkpoint)

## TODO: plot curves

## TODO: save metrics + plot

