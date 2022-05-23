
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x


class ImageSet(Dataset):
    def __init__(self, df, transformer,datasets):
        self.df = df
        self.transformer = transformer
        self.datasets=datasets
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_path = self.df.iloc[item]['img_path']
        if self.datasets!="mnist":
            image = self.transformer(Image.open(image_path).convert('RGB'))
        else:
            image = self.transformer(Image.open(image_path))
        label_idx = self.df.iloc[item]['label']
        suc = self.df.iloc[item]['not_suc']
        need_atk = self.df.iloc[item]['need_atk']
        sample = {
            'img': image,
            'lab': label_idx,
            'not_suc':suc,
            'need_atk':need_atk,
        }
        return sample

def mytest_loader( batch_size,dataframe,datasets,model_name='None'):
    if datasets=="imagenet":
        transformer = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor()])
        if model_name=="FBTF_Imagenet":
            transformer = transforms.Compose(
                [transforms.CenterCrop(288), transforms.ToTensor()])
    else:
        transformer = transforms.Compose([transforms.ToTensor()])
    datasets = {'test': ImageSet(dataframe, transformer, datasets)}
    dataloaders = DataLoader(datasets['test'],
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=False,
                             pin_memory=True)
    return dataloaders


class ImageSetMem(Dataset):
    def __init__(self, df, transformer,datasets,np_ds):
        self.df = df
        self.transformer = transformer
        self.datasets=datasets
        self.X,self.Y = np_ds
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        # image_path = self.df.iloc[item]['img_path']
        im_idx = self.df.iloc[item]['img_idx']
        label = self.df.iloc[item]['label']
        suc = self.df.iloc[item]['not_suc']
        need_atk = self.df.iloc[item]['need_atk']
        sample = {
            'img': self.transformer(self.X[im_idx,...]),
            'lab': label,
            'not_suc':suc,
            'need_atk':need_atk,
        }
        return sample



def mytest_loader_mem(batch_size,dataframe,datasets,np_ds,model_name='None'):
    if datasets=="imagenet":
        transformer = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor()])
        if model_name=="FBTF_Imagenet":
            transformer = transforms.Compose(
                [transforms.CenterCrop(288), transforms.ToTensor()])
    else:
        transformer = transforms.Compose([transforms.ToTensor()])
    datasets = {'test': ImageSetMem(dataframe, transformer, datasets, np_ds)}
    dataloaders = DataLoader(datasets['test'],
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=False,
                             pin_memory=True)
    return dataloaders


# margin loss
def margin_loss(logits,y,i=0,MT=True,device=None):
    bs = len(y)
    Y = y.view(-1,1)
    logit_org = logits.gather(1,Y)
    T = torch.eye(logits.shape[1])[y].to(device) * 999999
    TA = (logits-T)
    LTA = TA.argmax(1,keepdim=True)
    if MT:
        LTA_k = torch.topk(TA,TA.shape[-1],dim=1)
        LTA_s = torch.tensor([LTA_k[1][s][i%TA.shape[-1]] for s in range(bs)]).view(-1,1).to(device)
        logit_target = logits.gather(1,LTA_s)
    else:
        logit_target =  logits.gather(1,LTA)
    loss = -logit_org + logit_target
    return loss

def normalize_x(x):
    norm='L2'
    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    if norm == 'Linf':
        t = x.abs().view(x.shape[0], -1).max(1)[0]
        return x / (t.view(-1, *([1] * ndims)) + 1e-12)
    elif norm == 'L2':
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1, *([1] * ndims)) + 1e-12)

def lp_norm(x):
    norm ='L2'
    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    if norm == 'L2':
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return t.view(-1, *([1] * ndims))