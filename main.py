#torch
# import torchvision.transforms as transforms
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


## define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data', help='dataset directory path')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
# parser.add_argument('--hidden_size', type=int, default=60, help='hidden state size of gru module')
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate') #lr * lr_dc
# parser.add_argument('--lr_dc_step', type=int, default=45, help='the number of steps after which the learning rate decay') 
args = parser.parse_args()
print(args)

## set seeds + device
torch.manual_seed(522)
np.random.seed(522)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load data
## create dataset
## create dataloader
# actually using the validation split for initial shape tests
train_dataset = dataset.LoveDADataset(root_dir=args.root, split="train", transform=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# valid_dataset
# valid_loader
# test_dataset
# test_loader

## load splitted data

## define model (later will implement a CNN model)
model = models.deeplabv3_resnet50(pretrained=False)  
model.eval() 
#swap last layer for correct number of classes (7)
model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1)) #7 = num_class
# print(model.classifier)
model = model.to(device)

## define loss, optimizer, scheduler 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

## define variable for metrics

## loop data (epochs)
# remember to use device
for epoch in tqdm(range(args.epoch)):
    model.train()
    for i, (img, msk) in enumerate(tqdm(train_loader)):
        img, msk = img.to(device), msk.to(device)
        # print(img.size()) # (batch_size, channels, h, w) (10, 3, 129, 128)
        # print(msk.size())

        ## random samples for the mask 
        random_mask = random_points.create_random_points(msk, 0.1, 1) # using 10%
        random_mask = random_mask.squeeze(1).long().to(device=device)
        #random_points.plot_mask(random_mask)
        # print(random_mask)

        output = model(img)['out']
        # print(output.size()) # (batch_size, N_CLASSES, H, W)
        # print(random_mask.size())
        # print("hi")

        loss = criterion(output, random_mask)  
        loss.backward()
        optimizer.step()

        print(f"[TRAIN] epoch {epoch + 1}/{args.epoch} batch loss: {loss.item():.4f}")


    ## validation
    ## save checkpoint

## test model (load from best checkpoint)

## plot curves

## save metrics + plot

