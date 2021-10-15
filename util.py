import os, glob
import numpy as np

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%03d.pth" % (ckpt_dir, epoch))

def save_ps(ckpt_dir, net, optim, epoch, psnr, ssim, losses):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict(),
				'psnr': psnr,
                'ssim': ssim,
                'losses': losses},
               "%s/model_epoch%03d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def load_ps(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch, psnr, ssim, losses

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    psnr = dict_model['psnr']
    ssim = dict_model['ssim']

    return net, optim, epoch, psnr, ssim

def datasetListUp(data_dir, setsize):
	filelist = glob.glob(data_dir + '/*.npy')
	filelist = np.sort(filelist)
	datasetList = []
	for i in range(len(filelist) - setsize + 1):
		tmp = []
		for j in range(setsize):
			tmp.append(filelist[i+j])
		datasetList.append(tmp)
	return datasetList


def Psnr(label, pred):
  label = label.to('cpu').detach().numpy()
  pred = pred.to('cpu').detach().numpy()
  R = pred.max() if pred.max() > label.max() else label.max()
  return psnr(np.squeeze(label), np.squeeze(pred), data_range=R)

def Ssim(label, pred):
  print(pred.shape)
  label = label.to('cpu').detach().numpy().transpose(1,2,0)
  pred = pred.to('cpu').detach().numpy().transpose(1,2,0)
  rtn = 0
  for input1, input2 in zip(label, pred):
    rtn1 = ssim(input1, input2, multichannel=True)
    rtn += rtn1
  return rtn / len(label)
