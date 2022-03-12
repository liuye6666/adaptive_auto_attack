import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import  tqdm,trange
import pandas as pd
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import datetime
import math
import random
import time

# settings
device = torch.device("cuda")

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
    datasets = {'test': ImageSet(dataframe, transformer,datasets)}
    dataloaders = DataLoader(datasets['test'],
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=False,
                             pin_memory=True)
    return dataloaders

# margin loss
def margin_loss(logits,y,i=0,MT=True):
    bs = len(y)
    Y = y.view(-1,1)
    logit_org = logits.gather(1,Y)
    T = torch.eye(logits.shape[1])[y].to(device) * 999999
    TA = (logits-T)
    LTA = TA.argmax(1,keepdim=True)
    if MT:
        LTA_k = torch.topk(TA,TA.shape[-1],dim=1)
        LTA_s = torch.tensor([LTA_k[1][s][i%TA.shape[-1]] for s in range(bs)]).view(-1,1).to('cuda')
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

def AAA_white_box(model, X, X_adv, y, epsilon=0.031, step_num=20, ADI_step_num=8,
                  warm = False, bias_ODI=False, random_start = True, w_logit_dir=None, c_logit_dir=None,
                  sorted_logits=None, No_class = None, data_set = None, BIAS_ATK= None,
                  final_bias_ODI = False, No_epoch = 0, num_class=10, MT=False, Lnorm='Linf', out_re=None):
    tbn = 0 # number of  backward propagation
    tfn = 0 # number of  forward propagation
    if data_set=='mnist':
        threshold_class=9
    if data_set=="cifar10":
        threshold_class = 9
    if data_set=="cifar100":
        threshold_class =14
    if data_set=="imagenet":
        threshold_class =20
    if not warm:
        X_adv.data = X.data
    if random_start:
        random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
        X_adv = X_adv.data + random_noise
        X_adv = torch.clamp(X_adv, 0, 1.0)
        if warm:
            eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)
            X_adv = X.data + eta
            X_adv = torch.clamp(X_adv, 0, 1.0)

    atk_filed_index = torch.BoolTensor([1 for _ in range(len(y))]).to(device)
    np_atk_filed_index = np.ones(len(y)).astype(np.bool_)
    each_max_loss = -100000*np.ones(X.shape[0])
    total_odi_distribution =np.zeros((num_class,num_class+1))
    odi_atk_suc_num = 0
    each_back_num = 0

    randVector_ = torch.FloatTensor(X_adv.shape[0],num_class).uniform_(-1.0, 1.0).to(device)
    if bias_ODI and out_re>1:
        if BIAS_ATK:
            randVector_[np.arange(X_adv.shape[0]), y] = (num_class/10)*random.uniform(-0.1,0.5)
            no_r = -2 - No_class % threshold_class
            randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class/10)*0.8
        else:
            if w_logit_dir>0 and c_logit_dir>0:
                randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.1, 0.5)
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
            if w_logit_dir<0 and c_logit_dir>0:
                randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.1, 0.5)
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = -(num_class / 10) * 0.8
            if w_logit_dir<0 and c_logit_dir<0:
                randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.5, 0.1)
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = -(num_class / 10) * 0.8
            if w_logit_dir>=0 and c_logit_dir<=0:
                randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.5, 0.1)
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
            if data_set == "imagenet" or data_set == "cifar100":
                randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.5, 0.1)
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
        if final_bias_ODI:
            if BIAS_ATK:
                randVector_[np.arange(X_adv.shape[0]), y] = 0.0
                no_r = -2 - No_class % threshold_class
                randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class/10)*0.8
            else:
                if data_set=="cifar10" or data_set=="mnist":
                    final_threshold = 4
                    randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.8, 0.1)
                    no_r = -2 - No_class % final_threshold
                    randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
                if data_set=="cifar100":
                    final_threshold = 14
                    randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.8, 0.1)
                    no_r = -2 - No_class % final_threshold
                    randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
                if data_set=="imagenet":
                    final_threshold = 20
                    randVector_[np.arange(X_adv.shape[0]), y] = (num_class / 10) * random.uniform(-0.8, 0.1)
                    no_r = -2 - No_class % final_threshold
                    randVector_[np.arange(X_adv.shape[0]), sorted_logits[:, no_r]] = (num_class / 10) * 0.8
    tran_T2F=True
    when_atk_suc_iter_num = []
    for i in range(ADI_step_num + step_num):
        X_adv_f = X_adv[atk_filed_index,:,:,:]
        X_f = X[atk_filed_index,:,:,:]
        y_f = y[atk_filed_index]
        X_adv_f = Variable(X_adv_f,requires_grad=True)
        opt = optim.SGD([X_adv_f], lr=1e-3)
        each_back_num +=X_adv_f.shape[0]
        opt.zero_grad()
        with torch.enable_grad():
            if i< ADI_step_num:
                randVector_odi = randVector_[atk_filed_index, :]
                loss = (model(X_adv_f) * randVector_odi).sum()
                tfn+=X_adv_f.shape[0]
            else:
                if MT:
                    loss = margin_loss(model(X_adv_f), y[atk_filed_index],No_epoch,MT=MT).sum()
                else:
                    loss = margin_loss(model(X_adv_f), y[atk_filed_index]).sum()
                tfn+=X_adv_f.shape[0]
        loss.backward()
        tbn+=X_adv_f.shape[0]
        if i<ADI_step_num:
            if Lnorm =="Linf":
                eta = epsilon * X_adv_f.grad.data.sign()
            elif Lnorm=="L2":
                eta = epsilon * normalize_x(X_adv_f.grad.data)
        else:
            max_ssize = epsilon
            if Lnorm=='Linf':
                if step_num <=10:
                    min_ssize = epsilon* 0.1
                else:
                    min_ssize = epsilon* 0.001
            elif Lnorm=="L2":
                if step_num <=10:
                    min_ssize = epsilon* 0.1
                else:
                    min_ssize = epsilon* 0.001
            s = i - ADI_step_num
            step_size = min_ssize + (max_ssize - min_ssize) * (1 + math.cos(math.pi * s / (step_num + 1))) / 2
            if Lnorm == "Linf":
                eta = step_size * X_adv_f.grad.data.sign()
            elif Lnorm == "L2":
                eta = step_size * normalize_x(X_adv_f.grad.data)
        X_adv_f = X_adv_f.data + eta
        if Lnorm == 'Linf':
            eta = torch.clamp(X_adv_f.data - X_f.data, -epsilon, epsilon)
            X_adv_f = X_f.data + eta
            X_adv_f = torch.clamp(X_adv_f, 0, 1.0)
        elif Lnorm == 'L2':
            X_adv_f = torch.clamp(X_f + normalize_x(X_adv_f-X_f) *
                    torch.min(epsilon * torch.ones_like(X_f).detach(),lp_norm(X_adv_f - X_f)), 0.0, 1.0)

        output_ = model(X_adv_f)
        tfn+=X_adv_f.shape[0]
        output_loss = margin_loss(output_,y_f).squeeze()
        each_max_loss[np_atk_filed_index] = np.where(each_max_loss[np_atk_filed_index]>output_loss.detach().cpu().numpy(),
                                                     each_max_loss[np_atk_filed_index],output_loss.detach().cpu().numpy())
        X_adv[atk_filed_index,:,:,:]= X_adv_f
        _f_index = (output_.data.max(1)[1] == y_f.data)

        if out_re>=1:
            predict_wrong_num = int((~_f_index).float().sum().detach().cpu())
            # print(predict_wrong_num,out_re)
            if predict_wrong_num>0 and ADI_step_num>0:
                if tran_T2F:
                    tran_T2F = False
                    np_randVector_ = randVector_.detach().cpu().numpy().copy()
                    sort_np_rand = np_randVector_.copy()
                    for i in range(X_adv.shape[0]):
                        sort_np_rand[i, :] = np_randVector_[i, sorted_logits[i, :]]
                predict_wrong_class = (output_.data.max(1)[1][~_f_index]).detach().cpu().numpy()
                np_atk_filed_index=atk_filed_index.detach().cpu().numpy()
                for i in range(predict_wrong_num):
                    sort_np_rand_i = sort_np_rand[np_atk_filed_index, :][(~_f_index).cpu().numpy(),:][i,:]
                    total_odi_distribution[0, :num_class] += sort_np_rand_i
                    total_odi_distribution[0, num_class] += 1
                    atk_class = np.where(sorted_logits[np_atk_filed_index,:][(~_f_index).cpu().numpy(),:][i,:] == predict_wrong_class[i])[0].item()
                    total_odi_distribution[9-atk_class,:num_class] +=sort_np_rand_i
                    total_odi_distribution[9-atk_class,num_class] +=1

        atk_filed_index[atk_filed_index.clone()]=_f_index
        np_atk_filed_index = atk_filed_index.detach().cpu().numpy()
        if ADI_step_num > 0 and i==ADI_step_num-1:
            odi_atk_suc_num = atk_filed_index.shape[0] - atk_filed_index.float().sum()
    atk_acc_num = (model(X_adv).data.max(1)[1] == y.data).float().sum()
    tfn+=X_adv.shape[0]

    return atk_acc_num, np_atk_filed_index, X_adv, each_max_loss,odi_atk_suc_num,each_back_num,total_odi_distribution,tfn

def Adaptive_Auto_white_box_attack(model, device, eps, is_random, batch_size, average_num, model_name, data_set="cifar10",
                                   Lnorm='Linf'):
    ####  1.Loading datasets  ####
    if data_set=="mnist":
        num_class=10
        dataset_dir = 'data/mnist_test'
    if data_set=="cifar10":
        num_class = 10
        dataset_dir = 'data/cifar10/test'
    if data_set=="cifar100":
        num_class = 100
        dataset_dir = 'data/cifar100/test'
    if data_set=="imagenet":
        num_class = 1000
        dataset_dir = 'data/imagenet/test'
    if data_set=="cifar100" or data_set=="cifar10" or data_set=='mnist':
        all_atk_imgs = glob.glob(os.path.join(dataset_dir, '*/*.png'))
    else:
        all_atk_imgs = glob.glob(os.path.join(dataset_dir, '*/*.JPEG'))
    all_atk_labs = [int(img_path.split('/' )[-2]) for img_path in all_atk_imgs]
    not_suc_index = np.ones(len(all_atk_labs)).astype(np.bool)
    init_atk = pd.DataFrame({'img_path':all_atk_imgs,'label':all_atk_labs,'not_suc':not_suc_index,'need_atk':not_suc_index})
    data_loader = mytest_loader(batch_size=batch_size, dataframe=init_atk,datasets=data_set,model_name=model_name)
    total_odi_distribution = np.zeros((num_class,num_class+1))
    # 2.Initializing hyperparameters
    acc_curve = []
    robust_acc_oneshot = 0
    used_backward_num = 0
    total_adi_atk_suc_num = 0
    ori_need_atk_num = len(all_atk_labs)
    total_iter_num = ori_need_atk_num * average_num
    max_loss = -100000.0 * np.ones(ori_need_atk_num)
    logits_num = num_class
    sorted_logits_index = np.zeros((ori_need_atk_num,logits_num),dtype=np.int)
    natural_acc = 0
    begin_adv_acc = 0
    No_epoch = -2
    total_bn = 0
    total_fn = 0
    for i, test_data in enumerate(data_loader):
        bstart = i * batch_size
        bend = min((i + 1) * batch_size, len(not_suc_index))
        X, y = test_data['img'].to(device), test_data['lab'].to(device)
        out_clean = model(X)
        total_fn+=X.shape[0]
        acc_clean_num = (out_clean.data.max(1)[1] == y.data).float().sum()
        clean_filed_index = (out_clean.data.max(1)[1] == y.data).detach().cpu().numpy()
        natural_acc += acc_clean_num
        not_suc_index[bstart:bend] = not_suc_index[bstart:bend] * clean_filed_index
        if Lnorm =='Linf':
            random_noise = eps * torch.FloatTensor(*X.shape).uniform_(-eps, eps).sign().to(device)
            X_adv = X.data + random_noise
            X_adv = torch.clamp(X_adv, 0, 1.0)
        else:
            X_adv = X.data
        out_adv = model(X_adv)
        total_fn+=X_adv.shape[0]
        sorted_logits_index[bstart:bend] = np.argpartition(out_adv.detach().cpu().numpy(),
                                                           np.argmin(out_adv.detach().cpu().numpy(), axis=-1), axis=-1)[:, -logits_num:]
        acc_adv_num = (out_adv.data.max(1)[1] == y.data).float().sum()
        adv_filed_index = (out_adv.data.max(1)[1] == y.data).detach().cpu().numpy()
        begin_adv_acc += acc_adv_num
        not_suc_index[bstart:bend] = not_suc_index[bstart:bend] * adv_filed_index
    max_loss[~not_suc_index]= np.ones((~not_suc_index).sum())
    clean_acc = natural_acc/len(all_atk_labs)
    print(f"clean acc:{clean_acc:0.4}")

    out_restart_num = 13
    BIAS_ATK = False
    ADI = True
    MT=False
    wrong_logits_direct=0
    correct_logits_direct=0
    for out_re in trange(out_restart_num):
        restart_num = 1
        alpha = 1

        if out_re ==0:
            adi_iter_num = 0
            max_iter = 8
        else:
            adi_iter_num = 7
            max_iter = 8
        if out_re==1:
            restart_1_need_atk_num = robust_acc_oneshot
        if out_re==2:
            for i in range(1, num_class):
                wrong_logits_direct += total_odi_distribution[i,num_class-i-1]
                correct_logits_direct += total_odi_distribution[i,num_class-1]
            # print(total_odi_distribution)
            print(f"{model_name} ##################### wrong_dire:{wrong_logits_direct:0.4} "
                  f"correct_dire:{correct_logits_direct:0.4} ###################")
        if out_re >2:
            if total_adi_atk_suc_num<0.05 * restart_1_need_atk_num:
                BIAS_ATK = False
                if out_re>2:
                    alpha,max_iter = 0.7,25
                if out_re>3:
                    alpha,max_iter = 0.6,30
                if out_re>4:
                    alpha,max_iter = 0.5,30
                if out_re>5:
                    alpha,max_iter = 0.4,35
                if out_re>6:
                    alpha,max_iter = 0.3,40
                if out_re>7:
                    alpha,max_iter = 0.2,45
                if out_re>8:
                    alpha,max_iter = 0.1,50
                if out_re>9:
                    alpha, max_iter = 0.1, 50
                    restart_num = 4
                if out_re>10:
                    alpha, max_iter = 0.065, 55
                    restart_num = 4
                if out_re>11:
                    alpha, max_iter = 0.03, 60
                    restart_num = 50
                    if data_set=='imagenet':
                        restart_num=31
            else:
                BIAS_ATK = True
                if out_re>2:
                    alpha,max_iter,adi_iter_num = 1,5,10
                    restart_num = 10
                if out_re>3:
                    alpha,max_iter,adi_iter_num = 1,5,15
                    restart_num = 1
                if out_re>4:
                    alpha,max_iter,adi_iter_num = 1,10,15
                    restart_num = 1
                if out_re>5:
                    alpha,max_iter,adi_iter_num = 1,10,10
                    restart_num = 1
                if out_re>6:
                    alpha,max_iter,adi_iter_num = 1,15,10
                    restart_num = 1
                if out_re > 7:
                    alpha, max_iter, adi_iter_num = 1, 18, 10
                    restart_num = 1
                if out_re > 8:
                    alpha, max_iter, adi_iter_num = 1, 20, 10
                    restart_num = 1
                if out_re > 9:
                    alpha, max_iter, adi_iter_num = 0.8, 25, 10
                    restart_num = 5
                if out_re > 10:
                    alpha, max_iter, adi_iter_num = 0.5, 35, 10
                    restart_num = 5
                if out_re > 11:
                    alpha, max_iter, adi_iter_num = 0.5, 45, 10
                    restart_num = 20
        if data_set=="mnist":
            alpha=max(alpha,0.03)
        if data_set=="cifar10":
            alpha = max(alpha,0.03)
        if data_set=="cifar100":
            alpha = max(alpha,0.03)
        if data_set=="imagenet":
            alpha = max(alpha,0.03)
            max_iter =max_iter+10
            restart_num+=1
        if int(not_suc_index.sum())<len(not_suc_index)*0.1:
            alpha =1.0;restart_num +=10
        for r in range(restart_num):
            No_epoch+=1
            robust_acc_oneshot = 0

            remaining_iterations = total_iter_num - used_backward_num

            now_not_suc_num = int(not_suc_index.sum())
            need_atk_num_place1 = (len(not_suc_index)-now_not_suc_num)+min(max(100,int(now_not_suc_num*alpha)),now_not_suc_num)

            max_K = np.partition(max_loss, -need_atk_num_place1)[-need_atk_num_place1]
            loss_need_atk_index = (max_loss>=max_K) & (max_loss<0)
            not_suc_need_atk_index = loss_need_atk_index
            init_atk['not_suc'] = not_suc_index
            init_atk['need_atk'] = not_suc_need_atk_index
            now_sorted_logits_index = sorted_logits_index[not_suc_need_atk_index]
            fast_init_atk = init_atk.loc[init_atk['need_atk']==1]
            now_need_atk_num = not_suc_need_atk_index.sum()
            now_need_atk_index = np.ones(now_need_atk_num)
            now_max_loss = max_loss[not_suc_need_atk_index]
            data_loader = mytest_loader(batch_size=batch_size, dataframe=fast_init_atk,datasets=data_set,model_name=model_name)
            if (out_re<10 and BIAS_ATK==False) or(out_re<11 and BIAS_ATK==True):
                final_odi_atk = False
            else:
                final_odi_atk = True
            No_class = No_epoch
            iter_num = int(remaining_iterations/(restart_num*(out_restart_num-out_re))* now_need_atk_num)
            iter_num = min(max(iter_num,15),(max_iter+adi_iter_num))

            for i, test_data in (enumerate(data_loader)):
                bstart = i * batch_size
                bend = min((i + 1) * batch_size, now_need_atk_num)
                data, target = test_data['img'].to(device), test_data['lab'].to(device)
                sorted_logits = now_sorted_logits_index[bstart:bend]
                X, y = Variable(data, requires_grad=True), Variable(target)
                if True:warm=1
                else:warm=2
                for w in range(warm):
                    if w == 0:
                        X_input2 = X
                        WARM = False
                    else:
                        X_input2 = X_pgd
                        WARM = True
                    atk_filed_num_cold, atk_filed_index_cold, X_pgd, each_max_loss,odi_atk_suc_num,backward_num,each_odi_distribution,tfn = \
                        AAA_white_box(model, X, X_input2, y, eps, iter_num, adi_iter_num, WARM, ADI, is_random,
                                      sorted_logits=sorted_logits, No_class=No_class,w_logit_dir = wrong_logits_direct,c_logit_dir=correct_logits_direct,
                                      data_set=data_set, BIAS_ATK=BIAS_ATK,final_bias_ODI =final_odi_atk,No_epoch=No_epoch,num_class=num_class,
                                      MT=MT,Lnorm=Lnorm,out_re=out_re)
                    now_need_atk_index[bstart:bend] = now_need_atk_index[bstart:bend] * atk_filed_index_cold
                    now_max_loss[bstart:bend] = np.where(now_max_loss[bstart:bend]>each_max_loss,now_max_loss[bstart:bend],each_max_loss)
                    used_backward_num +=backward_num
                    total_adi_atk_suc_num+=odi_atk_suc_num
                    total_fn+=tfn
                    if out_re>=1:
                        total_odi_distribution +=each_odi_distribution
                robust_acc_oneshot += now_need_atk_index[bstart:bend].sum()

            not_suc_index[not_suc_need_atk_index]=now_need_atk_index
            max_loss[not_suc_need_atk_index]= now_max_loss

            print(f'No.{len(acc_curve)} clean_acc: {clean_acc:0.4} robust acc: {not_suc_index.sum()/len(all_atk_labs):0.4} used_bp_num: {used_backward_num} '
                  f'used_fp_num: {total_fn} now_atk_num: {now_need_atk_num} now_correct_num: {robust_acc_oneshot} ')
            acc_curve.append(f"{not_suc_index.sum()/len(all_atk_labs):0.4}")
            print(acc_curve)

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
# AWP
def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def level_sets_filter_state_dict(state_dict):
    from collections import OrderedDict
    if 'model_state_dict' in state_dict.keys():
        state_dict = state_dict['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'model.model.' in k:
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main(flag,model_name, ep=8./255,random=True, batch_size=128, average_number=100):

    stime = datetime.datetime.now()
    print(f"{flag} {model_name}")
    data_set="cifar10"
    Lnorm="Linf"
    if model_name=="TRADES":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        ep = 0.031
        model = WideResNet().to(device)
        model.load_state_dict(torch.load("model_weights/TRADES/TRADES_WRN.pt"))
    if model_name =="MART":
        from models.CIFAR10.MART_WRN import WideResNet
        model = WideResNet(depth=28).to(device)
        model = nn.DataParallel(model) # if widresnet_mart,we should use this line
        model.load_state_dict(torch.load("model_weights/MART/MART_UWRN.pt")['state_dict'])

    if model_name=="Feature_Scatter":
        from models.CIFAR10.Feature_Scatter import Feature_Scatter
        model = Feature_Scatter().to(device)
        model.load("model_weights/Feature_Scatter/Feature-Scatter")

    if model_name=="adv_inter":
        from models.CIFAR10.ADV_INTER.wideresnet import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10).to("cuda")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/ADV_INTER/latest")["net"])
        model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

    if model_name=="adv_regular":
        from models.CIFAR10.ADV_REGULAR.resnet import ResNet18
        model = ResNet18().to("cuda")
        model.load_state_dict(torch.load("model_weights/ADV_REGULAR/pretrained88.pth"))

    if model_name=="awp_28_10":
        from models.CIFAR10.AWP.wideresnet import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        ckpt = filter_state_dict(torch.load("model_weights/AWP/RST-AWP_cifar10_linf_wrn28-10.pt"))
        model.load_state_dict(ckpt)

    if model_name=="awp_34_10":
        from models.CIFAR10.AWP.wideresnet import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=10)
        ckpt = filter_state_dict(torch.load("model_weights/AWP/TRADES-AWP_cifar10_linf_wrn34-10.pt"))
        model.load_state_dict(ckpt)

    if model_name=="fbtf":
        from models.CIFAR10.FBTF.preact_resnet import PreActResNet18
        model = PreActResNet18().to("cuda")
        # model = nn.DataParallel(model)
        # model.load_state_dict(torch.load("model_weights/FBTF/cifar_model_weights_30_epochs.pth"),strict=False)
        model.load_state_dict(torch.load("model_weights/FBTF/Wong2020Fast.pt"),strict=False)
        model = nn.Sequential(Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]), model)

    if model_name=="geometry":
        from models.CIFAR10.GEOMETRY.wideresnet import WideResNet
        ep=0.031
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/GEOMETRY/Zhang2020Geometry.pt")["state_dict"])

    if model_name=="hydra":
        from models.CIFAR10.HYDRA.wrn_cifar import wrn_28_10
        model = wrn_28_10(nn.Conv2d, nn.Linear)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/HYDRA/model_best_dense.pth.tar", map_location="cuda")["state_dict"])

    if model_name=="hyer_embe":
        from models.CIFAR10.HYPER_EMBE.at_he import WideResNet
        model = WideResNet(widen_factor=20, use_FNandWN=True)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/HYPER_EMBE/model-wideres-pgdHE-wide20.pt")['state_dict'])

    if model_name=="level_sets":
        ep=0.031
        from models.CIFAR10.LEVEL_SETS.resnet import ResNet18
        model = ResNet18().to("cuda")
        ckpt = level_sets_filter_state_dict(torch.load("model_weights/LEVEL_SETS/200.pth"))
        model.load_state_dict(ckpt)

    if model_name=="mma":
        # from models.CIFAR10.MMA.MMA import WideResNet
        # model = WideResNet(depth=28, widen_factor=4, sub_block1=False).to("cuda")
        from models.CIFAR10.MMA.MMA import Ding2020MMANet
        model = Ding2020MMANet().to("cuda")
        ckpt = torch.load("model_weights/MMA/Ding2020MMA.pt")
        model.load_state_dict(ckpt)
        # model = nn.Sequential(Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]), model)

    if model_name=="overfit":
        from models.CIFAR10.OVERFIT.robust_overfitting import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=20)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/OVERFIT/cifar10_wide20_linf_eps8.pth"))
        model = nn.Sequential(Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]), model)

    if model_name=="pre_train":
        from models.CIFAR10.PRE_TRAIN.pre_training import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10).to("cuda")
        model = nn.DataParallel(model)
        model.module.fc = nn.Linear(640, 10)
        model.load_state_dict(torch.load("model_weights/PRE_TRAIN/cifar10wrn_baseline_epoch_4.pt"))
        model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

    if model_name=="proxy_dist":
        from models.CIFAR10.OVERFIT.robust_overfitting import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=10)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/PROXY_DIST/Sehwag2021Proxy.pt")["state_dict"])

    if model_name=="robustness":
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import CIFAR
        ds = CIFAR('/path/to/cifar')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                                          resume_path='model_weights/ROBUSTNESS/cifar_linf_8.pt')

    if model_name=="rst":
        from models.CIFAR10.RST.rst import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/RST/rst_adv.pt.ckpt")["state_dict"])

    if model_name=="self_adaptive":
        ep=0.031
        from models.CIFAR10.SELF_ADAPTIVE.wideresnet import wrn34
        model = wrn34(num_classes=10).to("cuda")
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/self-adaptive/model-wideres-epoch78.pth"))
    if model_name=="sensible":
        from models.CIFAR10.SENSIBLE.wideresnet import WideResNet
        model = WideResNet().to("cuda")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/SENSIBLE/SENSE_checkpoint300.dict")["state_dict"])

    if model_name=="understanding_fast":
        from models.CIFAR10.Understanding_FAST.preactresnet import PreActResNet18
        model = PreActResNet18(10).to("cuda")
        model.load_state_dict(
            torch.load("model_weights/Understanding_FAST/Andriushchenko2020Understanding.pt")["last"])

    if model_name=="yopo":
        from models.CIFAR10.YOPO.wideresnet import WideResNet
        model = WideResNet(depth=34)
        model.load_state_dict(torch.load("model_weights/YOPO/Zhang2019You.pt", map_location="cuda"))

    if model_name=="ULAT_28_10_with":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=28, width=10,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/cifar10_linf_wrn28-10_with.pt"))

    if model_name=="ULAT_70_16_extra":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/Gowal2020Uncovering_70_16_extra.pt"))
        batch_size=32

    if model_name=="ULAT_34_20":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=34, width=20,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/cifar10_linf_wrn34-20_without.pt"))
        batch_size=64

    if model_name=="ULAT_70_16":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/cifar10_linf_wrn70-16_without.pt"))
        batch_size=32

    if model_name=="fix_data_28_10_with":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=28, width=10,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar10_linf_wrn28-10_cutmix_ddpm_v2.pt"))

    if model_name=="fix_data_70_16_extra":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar10_linf_wrn70-16_cutmix_ddpm_v2.pt"))
        batch_size=32

    if model_name=="IAR":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        model = WideResNet().to(device)
        model.load_state_dict(torch.load("model_weights/IAR/cifar10_wrn.pt"))

    if model_name=="LBGAT_34_10":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        ep = 0.031
        model = WideResNet().to(device)
        model.load_state_dict(torch.load("model_weights/LBGAT/cifar10_lbgat0_wideresnet34-10.pt"))

    if model_name=="LBGAT_34_20":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        ep = 0.031
        model = WideResNet(widen_factor=20).to(device)
        model.load_state_dict(torch.load("model_weights/LBGAT/cifar10_lbgat0_wideresnet34-20.pt"))

    if model_name=="FAT":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        model = WideResNet().to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/FAT/fat_for_trades_wrn34-10_eps0.062_beta6.0.pth.tar")["state_dict"])

    if model_name=="proxy_dist_r18":
        from models.CIFAR10.PROXY_DIST.resnet import ResNet18
        model = ResNet18().to("cuda")
        # model = torch.nn.DataParallel(model)
        ckpt = (torch.load("model_weights/PROXY_DIST/Sehwag2021Proxy_R18.pt"))
        model.load_state_dict(ckpt)
    if model_name=="TRPF":
        from models.CIFAR10.TRPF.resnet import ResNet18
        from models.CIFAR10.TRPF.normalization_layer import Normalize_layer
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        model = ResNet18(num_classes=10).to("cuda")
        model = torch.nn.Sequential(
            Normalize_layer(mean, std),
            model)
        # model = torch.nn.DataParallel(model)
        ckpt = (torch.load("model_weights/TRPF/resnet18_cifar10_dens0.05_magnitude_epoch200_testAcc_87.31999969482422.pt"))
        model.load_state_dict(ckpt)

    if model_name=="OAAT_r18":
        from models.CIFAR10.OAAT.resnet import ResNet18
        model = ResNet18(num_classes=10).to("cuda")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR10_RN18.pt"))

    if model_name=="OAAT_wrn34":
        from models.CIFAR10.OAAT.widresnet import WideResNet # TRADES_WRN
        model = WideResNet(num_classes=10).to("cuda")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR10_WRN34.pt"))

    if model_name=="RLPE_28_10":
        from models.CIFAR10.RST.rst import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/RLPE/RST_0.1485.pt")["state_dict"])

    if model_name=="RLPE_34_15":
        from models.CIFAR10.RST.rst import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=15)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/RLPE/Wide-RST_0.1485.pt")["state_dict"])

    if model_name=="ULAT_70_16_with_100":
        data_set = "cifar100"
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=100, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
            std=widresnet.CIFAR100_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/cifar100_linf_wrn70-16_with.pt"))
        batch_size=32

    if model_name=="ULAT_70_16_100":
        data_set = "cifar100"
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=100, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
            std=widresnet.CIFAR100_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/cifar100_linf_wrn70-16_without.pt"))
        batch_size=32

    if model_name=="fix_data_28_10_with_100":
        data_set = "cifar100"
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=100, depth=28, width=10,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
            std=widresnet.CIFAR100_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar100_linf_wrn28-10_cutmix_ddpm.pt"))

    if model_name=="fix_data_70_16_extra_100":
        data_set = "cifar100"
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=100, depth=70, width=16,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
            std=widresnet.CIFAR100_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar100_linf_wrn70-16_cutmix_ddpm.pt"))
        batch_size=32

    if model_name=="OAAT_r18_100":
        data_set = "cifar100"
        from models.CIFAR10.OAAT.preactresnet import PreActResNet18
        model = PreActResNet18(num_classes=100).to("cuda")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR100_PRN18.pkl"))

    if model_name=="OAAT_wrn34_100":
        data_set = "cifar100"
        from models.CIFAR10.OAAT.widresnet import WideResNet # TRADES_WRN
        model = WideResNet(num_classes=100).to("cuda")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR100_WRN34.pkl"))

    if model_name=="LBGAT_34_10_100":
        data_set = "cifar100"
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        ep = 0.031
        model = WideResNet(num_classes=100).to(device)
        model.load_state_dict(torch.load("model_weights/LBGAT/cifar100_lbgat6_wideresnet34-10.pt"))

    if model_name=="LBGAT_34_20_100":
        data_set = "cifar100"
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        ep = 0.031
        model = WideResNet(num_classes=100,widen_factor=20).to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/LBGAT/cifar100_lbgat6_wideresnet34-20.pt"))

    if model_name=="awp_34_10_100":
        data_set = "cifar100"
        from models.CIFAR10.AWP.wideresnet import WideResNet
        model = WideResNet(depth=34, num_classes=100, widen_factor=10)
        ckpt = filter_state_dict(torch.load("model_weights/AWP/AT-AWP_cifar100_linf_wrn34-10.pth"))
        model.load_state_dict(ckpt)
        model = nn.Sequential(Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]), model)

    if model_name=="pre_train_28_10_100":
        data_set = "cifar100"
        from models.CIFAR10.PRE_TRAIN.pre_training import WideResNet
        model = WideResNet(depth=28, num_classes=100, widen_factor=10).to("cuda")
        model = nn.DataParallel(model)
        model.module.fc = nn.Linear(640, 100)
        model.load_state_dict(torch.load("model_weights/PRE_TRAIN/cifar100wrn_baseline_epoch_4.pt"))
        model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

    if model_name=="IAR_100":
        data_set = "cifar100"
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        model = WideResNet(num_classes=100).to(device)
        model.load_state_dict(torch.load("model_weights/IAR/cifar100_wrn.pt"))

    if model_name=="overfit_100":
        data_set = "cifar100"
        from models.CIFAR10.OVERFIT.preactresnet import PreActResNet18
        model = PreActResNet18(num_classes=100)
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/OVERFIT/cifar100_linf_eps8.pth"))
        CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        model = nn.Sequential(Normalize(CIFAR100_MEAN, CIFAR100_STD), model)

    if model_name =="Salman2020Do_R18":
        data_set = 'imagenet'
        ep=4/255
        from torchvision import models as pt_models
        model = pt_models.resnet18().to("cuda")
        model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_R18.pt"))
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
        batch_size=64

    if model_name =="Salman2020Do_R50":
        data_set = 'imagenet'
        ep=4/255
        from torchvision import models as pt_models
        model = pt_models.resnet50().to("cuda")
        model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_R50.pt"))
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
        batch_size=64

    if model_name =="Salman2020Do_50_2":
        data_set = 'imagenet'
        ep=4/255
        from torchvision import models as pt_models
        model = pt_models.wide_resnet50_2().to("cuda")
        model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_50_2.pt"))
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
        batch_size=32

    if model_name =="FBTF_Imagenet":
        data_set = 'imagenet'
        ep=4/255
        from torchvision import models as pt_models
        model = pt_models.resnet50().to("cuda")
        model.load_state_dict(torch.load("model_weights/FBTF/Wong2020Fast_I.pt"))
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
        batch_size=32

    if model_name=="proxy_dist_L2":
        from models.CIFAR10.OVERFIT.robust_overfitting import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=10)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/PROXY_DIST/Sehwag2021ProxyL2.pt")["state_dict"])
        Lnorm='L2'
        batch_size=128
        ep=0.5

    if model_name=="overfit_R18_L2":
        data_set = "cifar10"
        from models.CIFAR10.OVERFIT.preactresnet import PreActResNet18
        model = PreActResNet18(num_classes=10)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("model_weights/OVERFIT/Rice2020OverfittingL2.pt"))
        model = nn.Sequential(Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]), model)
        Lnorm='L2'
        batch_size=128
        ep=0.5

    if model_name=="fix_data_28_10_L2":
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=28, width=10,
            activation_fn=widresnet.Swish, mean=widresnet.CIFAR10_MEAN,
            std=widresnet.CIFAR10_STD)
        model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar10_l2_wrn28-10_cutmix_ddpm_v2.pt"))
        Lnorm='L2'
        batch_size=128
        ep=0.5

    if model_name=="robustness_L2":
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import CIFAR
        ds = CIFAR('/path/to/cifar')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                                          resume_path='model_weights/ROBUSTNESS/cifar_l2_0_5.pt')
        Lnorm='L2'
        batch_size=128
        ep=0.5

    if model_name=="DARI_densenet_L2":
        data_set = 'imagenet'
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import ImageNet
        from torchvision import models
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch=models.densenet161(), dataset=ds,
                                          resume_path='model_weights/DARI/densenet_l2_eps3.ckpt',
                                          add_custom_forward=True)
        Lnorm='L2'
        batch_size=12
        ep=3.0

    if model_name=="DARI_VGG16_L2":
        data_set = 'imagenet'
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import ImageNet
        from torchvision import models
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch=models.vgg16_bn(), dataset=ds,
                                          resume_path='model_weights/DARI/vgg16_bn_l2_eps3.ckpt',
                                          parallel=False)
        Lnorm='L2'
        batch_size=16
        ep=3.0

    if model_name == "DARI_ShuffleNet_L2":
        data_set = 'imagenet'
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import ImageNet
        from torchvision import models
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch=models.shufflenet_v2_x1_0(), dataset=ds,
                                          resume_path='model_weights/DARI/shufflenet_l2_eps3.ckpt',
                                          parallel=False)
        Lnorm = 'L2'
        batch_size = 16
        ep = 3.0

    if model_name == "DARI_mobilenet_L2":
        data_set = 'imagenet'
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import ImageNet
        from torchvision import models
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch=models.mobilenet_v2(), dataset=ds,
                                          resume_path='model_weights/DARI/mobilenet_l2_eps3.ckpt',
                                          parallel=False)
        Lnorm = 'L2'
        batch_size = 16
        ep = 3.0

    if model_name=="TRADES_mnist":
        data_set='mnist'
        from models.mnist.small_cnn import SmallCNN
        model= SmallCNN()
        model.load_state_dict(torch.load("model_weights/TRADES/model_mnist_smallcnn.pt"))
        ep=0.3
        batch_size=512

    if model_name=="ULAT_mnist":
        data_set='mnist'
        from models.CIFAR10.FIX_DATA import widresnet
        model_ctor = widresnet.WideResNet
        model = model_ctor(
            num_classes=10, depth=28, width=10,
            activation_fn=widresnet.Swish, mean=0.5,
            std=0.5, padding=2, num_input_channels=1)
        model.load_state_dict(torch.load("model_weights/FIX_DATA/mnist_linf_wrn28-10_without.pt"))
        batch_size=128
        ep=0.3

    if model_name =="Standard_imagenet":
        data_set = 'imagenet'
        ep=4/255
        from torchvision import models as pt_models
        model = pt_models.resnet50(pretrained=True).to("cuda")
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
        batch_size=64

    if model_name=="robustness_imagenet":
        data_set="imagenet"
        ep=4/255
        from robustness.model_utils import make_and_restore_model
        from robustness.datasets import CIFAR,ImageNet
        ds = ImageNet('/path/to/cifar')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                                          resume_path='model_weights/ROBUSTNESS/imagenet_linf_4.pt')
        batch_size=64


    model.to("cuda")
    model.eval()
    Adaptive_Auto_white_box_attack(model, device, ep, random, batch_size, average_number, model_name, data_set=data_set,Lnorm=Lnorm)
    etime = datetime.datetime.now()
    print(f'use time:{etime-stime}s')

if __name__ == '__main__':

    # IAR,LBGAT_34_10,LBGAT_34_20,FAT,proxy_dist_r18,TRPF,OAAT_r18,OAAT_wrn34,RLPE_28_10,RLPE_34_15,ULAT_70_16_with_100
    # main('AAA',model_name="OAAT_r18", average_number=1000)
    # main('AAA',model_name="IAR", average_number=1000)
    # main('AAA',model_name="LBGAT_34_10", average_number=1000)
    # main('AAA',model_name="LBGAT_34_20", average_number=1000)
    # main('AAA',model_name="FAT", average_number=1000)
    # main('AAA',model_name="proxy_dist_r18", average_number=1000)
    # main('AAA',model_name="TRPF", average_number=1000)
    # main('AAA',model_name="OAAT_wrn34", average_number=1000)
    # main('AAA',model_name="RLPE_28_10", average_number=1000)
    # main('AAA',model_name="RLPE_34_15", average_number=1000)
    # main('AAA',model_name="ULAT_70_16_with_100", average_number=1000)
    # robustness,rst,self_adaptive,sensible,understanding_fast,yopo,ULAT_28_10_with,ULAT_70_16_extra,ULAT_34_20,ULAT_70_16,fix_data_28_10_with,fix_data_70_16_extra
    # main('AAA',model_name="robustness", average_number=1000)
    # main('AAA',model_name="rst", average_number=1000)
    # main('AAA',model_name="self_adaptive", average_number=1000)
    # main('AAA',model_name="sensible", average_number=1000)
    # main('AAA',model_name="understanding_fast", average_number=1000)
    # main('AAA',model_name="yopo", average_number=1000)
    # main('AAA',model_name="ULAT_28_10_with", average_number=1000)
    # main('AAA',model_name="ULAT_70_16_extra", average_number=1000)
    # main('AAA',model_name="ULAT_34_20", average_number=1000)
    # main('AAA',model_name="ULAT_70_16", average_number=1000)
    # main('AAA',model_name="fix_data_28_10_with", average_number=1000)
    #main('AAA',model_name="fix_data_70_16_extra", average_number=1000)
    # # ULAT_70_16_100,fix_data_28_10_with_100,fix_data_70_16_extra_100,OAAT_r18_100,OAAT_wrn34_100,LBGAT_34_10_100,LBGAT_34_20_100
    # main('AAA',model_name="ULAT_70_16_100", average_number=1000)
    # main('AAA',model_name="fix_data_28_10_with_100", average_number=1000)
    # main('AAA',model_name="fix_data_70_16_extra_100", average_number=1000)
    # main('AAA',model_name="OAAT_r18_100", average_number=1000)
    # main('AAA',model_name="OAAT_wrn34_100", average_number=1000)
    # main('AAA',model_name="LBGAT_34_10_100", average_number=1000)
    # main('AAA',model_name="LBGAT_34_20_100", average_number=1000)
    # # TRADES,MART,Feature_Scatter,adv_inter,adv_regular,awp_28_10,awp_34_10,fbtf,geometry,hydra,hyer_embe,level_sets,mma,overfit,pre_train,proxy_dist
    # main('AAA',model_name="TRADES", average_number=500)
    # main('AAA',model_name="MART", average_number=1000)
    # main('AAA',model_name="Feature_Scatter", average_number=1000)
    # main('AAA',model_name="adv_inter", average_number=1000)
    # main('AAA',model_name="adv_regular", average_number=1000)
    # main('AAA',model_name="awp_28_10", average_number=1000)
    # main('AAA',model_name="awp_34_10", average_number=1000)
    #main('AAA',model_name="fbtf", average_number=1000)
    # main('AAA',model_name="geometry", average_number=1000)
    # main('AAA',model_name="hydra", average_number=1000)
    # main('AAA',model_name="hyer_embe", average_number=1000)
    # main('AAA',model_name="level_sets", average_number=1000)
    # main('AAA',model_name="mma", average_number=1000)
    # main('AAA',model_name="overfit", average_number=1000)
    # main('AAA',model_name="pre_train", average_number=1000)
    # main('AAA',model_name="proxy_dist", average_number=1000)
    # awp_34_10_100,pre_train_28_10_100,IAR_100,overfit_100,Salman2020Do_R18,Salman2020Do_R50,Salman2020Do_50_2,FBTF_Imagenet
    # main('AAA',model_name="awp_34_10_100", average_number=1000)
    # main('AAA',model_name="pre_train_28_10_100", average_number=1000)
    # main('AAA',model_name="IAR_100", average_number=1000)
    # main('AAA',model_name="overfit_100", average_number=1000)
    # main('AAA',model_name="Salman2020Do_R18", average_number=1000)
    # main('AAA',model_name="Salman2020Do_R50", average_number=1000)
    #main('AAA',model_name="Salman2020Do_50_2", average_number=1000)
    #main('AAA',model_name="FBTF_Imagenet", average_number=1000)
    # # proxy_dist_L2,overfit_R18_L2,fix_data_28_10_L2,robustness_L2,DARI_densenet_L2,DARI_VGG16_L2,DARI_ShuffleNet_L2
    # main('AAA',model_name="proxy_dist_L2", average_number=1000)
    # main('AAA',model_name="overfit_R18_L2", average_number=1000)
    # main('AAA',model_name="fix_data_28_10_L2", average_number=1000)
    # main('AAA',model_name="robustness_L2", average_number=1000)
    # main('AAA',model_name="DARI_densenet_L2", average_number=1000)
    # main('AAA',model_name="DARI_VGG16_L2", average_number=1000)
    # main('AAA',model_name="DARI_ShuffleNet_L2", average_number=1000)
    # # DARI_mobilenet_L2,TRADES_mnist,ULAT_mnist,Standard_imagenet,robustness_imagenet
    # main('AAA',model_name="DARI_mobilenet_L2", average_number=1000)
    main('AAA',model_name="TRADES_mnist", average_number=1000)
    # main('AAA',model_name="ULAT_mnist", average_number=1000)
    # main('AAA',model_name="Standard_imagenet", average_number=1000)
    # main('AAA',model_name="robustness_imagenet", average_number=1000)


