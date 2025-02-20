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
import csv   
import time
from datetime import datetime

#local 
from utils import dataset, random_points, metrics
from loss import partial_cross_entropy

## define args
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)  
parser.add_argument('--mask_percentage', type=float, default=0.3)  
parser.add_argument('--resize_to', type=int, default=256)  
parser.add_argument('--data_aug', type=int, default=0)  
args = parser.parse_args()
print(args)

## set seeds + device
torch.manual_seed(522)
np.random.seed(522)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load data + create dataset + create dataloader
train_dataset = dataset.LoveDADataset(root_dir=args.root, resize_to=args.resize_to, split="train", transform=True)
valid_dataset = dataset.LoveDADataset(root_dir=args.root, resize_to=args.resize_to,split="valid", transform=True)
#test_dataset = dataset.LoveDADataset(root_dir=args.root, resize_to=args.resize_to,split="test", transfaaorm=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


## define model (TODO: later will implement a CNN model)
model = models.deeplabv3_resnet50(pretrained=True)  
# model.eval() 
#swap last layer for correct number of classes # 7 = num_class
model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1)) 
model = model.to(device)
# print(model.classifier)

## define loss, optimizer #criterion = nn.CrossEntropyLoss()
criterion = partial_cross_entropy.PartialCrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

## TODO: define variable for metrics (right now will use only loss)
best_valid_loss = float('inf')
best_iou = 0
losses, valid_losses = [], []
timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
start_time = time.time()


def validate(loader, model, criterion):
    model.eval()
    sum_valid_loss = 0
    ious = []

    with torch.no_grad():
        for i, (img, msk) in enumerate(tqdm(loader)):
            img, msk = img.to(device), msk.to(device)
            random_mask = random_points.create_random_points(msk, 0.3, 1) 
            random_mask = random_mask.squeeze(1).long().to(device=device)
            output = model(img)['out']
            loss = criterion(output, random_mask)  
            sum_valid_loss += loss.item()
            ious.append(metrics.Iou(random_mask, output))

    return (sum_valid_loss / len(loader)), torch.mean(ious)


for epoch in tqdm(range(args.epoch)):
    model.train()

    sum_epoch_loss = 0
    for i, (img, msk) in enumerate(tqdm(train_loader)):
        img, msk = img.to(device), msk.to(device)
        # print(img.size()) # (batch_size, channels, h, w) (10, 3, 129, 128)
        # print(msk.size())

        ## random samples for the mask 

        # there was a 0 class, check what happened to that. 
        random_mask = random_points.create_random_points(msk, args.mask_percentage, 1) 
        random_mask = random_mask.squeeze(1).long().to(device=device)
        # random_points.plot_mask(random_mask, 0)
        # random_points.plot_mask(random_mask, 1)
        # print(random_mask)

        output = model(img)['out']
        # print(output.size()) # (batch_size, N_CLASSES, H, W)
        # print(random_mask.size())
        # print("hi")

        loss = criterion(output, random_mask)  
        sum_epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[TRAIN] epoch {epoch + 1}/{args.epoch} batch loss: {loss.item():.4f}")

    losses.append(sum_epoch_loss/len(train_loader))

    ## TODO: validation loss
    valid_loss, iou = validate(valid_loader, model, criterion)
    valid_losses.append(valid_loss)


    ## save checkpoint
    if valid_loss < best_valid_loss:
        best_iou = iou
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"checkpoints/{timestamp}_test.pth")

## plot curves + save curve + metrics (?)
plt.clf()
plt.plot(losses,label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training/Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.savefig(f'loss_curves/{timestamp}_loss_curve.png')

fields=[timestamp, best_valid_loss, args.batch_size, args.epoch, args.lr, args.mask_percentage, args.resize_to, args.data_aug, (time.time() - start_time), best_iou]  
with open(r'stats/data.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

## TODO: test model (load from best checkpoint)
#model.load_state_dict(torch.load(f"checkpoints/test.pth"))
#test_loss = validate(test_loader, model, criterion)
# Visualize images/inference



