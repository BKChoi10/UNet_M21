import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Seq_Dataset(Dataset):

  def __init__(self,datasets,channel=None):
    self.datasets = datasets
    self.channel = channel
    self.total_dataset = []
    
    for dataset in datasets:
      arange = np.arange(len(dataset))
      dset = []
      for i in (arange):
        dset.append(dataset[i])
      self.total_dataset.append(dset)

  def __getitem__(self, idx):
    da = self.total_dataset[idx]
    datas = [np.load(d)[:,:,self.channel] for d in da[:-1]]
    images = [np.array(data) for data in datas]
    images = np.stack([[image] for image in images])
    images = np.squeeze(images)
    mean = [image.mean() for image in images]
    std = [image.std() for image in images]
    images = images.transpose(1,2,0)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    input_image = transform(images)
    output_image = np.load(da[-1])[:,:,self.channel]
    output_image = torch.as_tensor(output_image, dtype=torch.float32)

    return input_image, output_image

  def __len__(self):
    return len(self.total_dataset)
