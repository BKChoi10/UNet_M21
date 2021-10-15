import numpy as np
import torch
import torch.nn as nn
import glob, os, datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from Seq_Dataset import Seq_Dataset
from unet import UNet_residual
from util import *

#Initialize
train_dir = './dataset/train'
val_dir = './dataset/val'
test_dir = './dataset/test'
result_dir = './result'
ckpt_dir = './checkpoint'
lr = 1e-5
train_continue = 'on'

if not os.path.exists(result_dir):
  os.makedirs(result_dir)
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)

print('Run..')
print('train_dir :' + train_dir)
print('result_dir :' + result_dir)
print('ckpt_dir :' + ckpt_dir)
print('lr :', lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Dataload
datasetList = datasetListUp(train_dir,13)
train_dataset = Seq_Dataset(datasets=datasetList,channel=10)
train_loader = DataLoader(dataset = train_dataset, batch_size=48, shuffle=True)

datasetList = datasetListUp(val_dir,13)
val_dataset = Seq_Dataset(datasets=datasetList,channel=10)
val_loader = DataLoader(dataset = val_dataset, batch_size=48, shuffle=False)


#Additional
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)

#Model
_net = UNet_residual(n_channels=12,n_classes=1).to(device)
net = nn.DataParallel(_net).to(device)

#Loss function
loss = nn.MSELoss().to(device)

#Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

num_epoch = 500
st_epoch = 0
Best_psnr = 0
Best_ssim = 0

#Load
if train_continue == "on":
  net, optimizer, st_epoch, Best_psnr, Best_ssim = load_ps(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

now = datetime.datetime.now()
print_out = now.strftime('%Y-%m-%d %H:%M')
flog = open("log", 'a')
flog.write('Run Start')
flog.write(print_out + '\n')
flog.close()
#Train
for epoch in range(st_epoch + 1, num_epoch + 1):
  net.train()
  train_loss = 0
  train_psnr = 0
  train_ssim = 0
  for batch_idx, samples in enumerate(train_loader):
    x_train1, y_train = samples
    x_train1 = x_train1.to(device)
    y_train = y_train.to(device)
    prediction = net(x_train1.float())
    prediction = np.squeeze(prediction.float())
    y_train = np.squeeze(y_train.float())
    cost = loss(prediction, y_train)
    train_psnr1 = Psnr(y_train, prediction)
    train_ssim1 = Ssim(y_train, prediction)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    train_loss += cost.item()
    train_psnr += train_psnr1
    train_ssim += train_ssim1
#    print_out='Train : Epoch {:4d}/{} Batch {}/{} Cost: {:6f} PSNR: {:6f} SSIM: {:6f}\n'.format(
#    epoch, num_epoch, batch_idx+1, len(train_loader), train_loss / (batch_idx+1), train_psnr / (batch_idx+1), train_ssim / (batch_idx+1)
#    )
#    print(print_out)
  print_out='Train : Epoch {:4d}/{} Batch {}/{} Cost: {:6f} PSNR: {:6f} SSIM: {:6f}\n'.format(
  epoch, num_epoch, batch_idx+1, len(train_loader), train_loss / (batch_idx+1), train_psnr / (batch_idx+1), train_ssim / (batch_idx+1)
  )
  flog = open("log", 'a')
  flog.write(print_out)
  flog.close()
  train_psnr_mean  = train_psnr / len(train_loader)
  train_ssim_mean  = train_ssim / len(train_loader)
  #if train_psnr_mean > Best_psnr and train_ssim_mean > Best_ssim :
  #  Best_psnr = train_psnr_mean
  #  Best_ssim = train_ssim_mean
  #  save_ps(ckpt_dir=ckpt_dir, net=net, optim=optimizer, epoch=epoch, psnr=Best_psnr, ssim=Best_ssim, losses=(train_loss / len(train_loader)))
  save_ps(ckpt_dir=ckpt_dir, net=net, optim=optimizer, epoch=epoch, psnr=train_psnr_mean, ssim=train_ssim_mean, losses=(train_loss / len(train_loader)))

  with torch.no_grad():
    net.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    for j, val in enumerate(val_loader):
        x_val, y_val = val
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        val_output = net(x_val.float())
        val_output = np.squeeze(val_output.float())
        y_val = np.squeeze(y_val.float())
        v_loss = loss(val_output, y_val)
        v_psnr = Psnr(y_val, val_output)
        v_ssim = Ssim(y_val, val_output)
        val_loss += v_loss
        val_psnr += v_psnr
        val_ssim += v_ssim
    print_out='Val : Epoch {:4d}/{} Batch {}/{} Cost: {:6f} PSNR: {:6f} SSIM: {:6f}\n'.format(
        epoch, num_epoch, j+1, len(val_loader), val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader)
    )
#    print(print_out)
    flog = open("log", 'a')
    flog.write(print_out)
    flog.close()
    val_psnr_mean  = val_psnr / len(val_loader)
    val_ssim_mean  = val_ssim / len(val_loader)
  now = datetime.datetime.now()
  print_out = now.strftime('%Y-%m-%d %H:%M')
  flog = open("log", 'a')
  flog.write(print_out + '\n')
  flog.close()
#    if val_psnr_mean > Best_psnr and val_ssim_mean > Best_ssim :
#      Best_psnr = val_psnr_mean
#      Best_ssim = val_ssim_mean
#      save_ps(ckpt_dir=ckpt_dir, net=net, optim=optimizer, epoch=epoch, psnr=Best_psnr, ssim=Best_ssim, losses=(val_loss / len(val_loader)))
