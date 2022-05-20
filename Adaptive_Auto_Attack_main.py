import os
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
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# global device
# device = torch.device("cuda")

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

def AAA_white_box(model, X, X_adv, y, epsilon=0.031, step_num=20, ADI_step_num=8,
                  warm = False, bias_ODI=False, random_start = True, w_logit_dir=None, c_logit_dir=None,
                  sorted_logits=None, No_class = None, data_set = None, BIAS_ATK= None,
                  final_bias_ODI = False, No_epoch = 0, num_class=10, MT=False, Lnorm='Linf', out_re=None,
                  device=None,neptune_run=None, **kwargs):
    print('AAA_white_box',device)
    # device = kwargs['device']
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

    try:
        p = kwargs['normal_prob']
        choice_ = np.random.choice([0,1],p=[1-p,p])
        if choice_ == 1:
            randVector_ = torch.FloatTensor(X_adv.shape[0],num_class).normal_(0.0, .5).to(device)

        else:
            randVector_ = torch.FloatTensor(X_adv.shape[0],num_class).uniform_(-1.0, 1.0).to(device)

    except KeyError:
        randVector_ = torch.FloatTensor(X_adv.shape[0],num_class).uniform_(-1.0, 1.0).to(device)


    #
    #[BEN Comment] this is where they are adaptively changing the direction of the initilizaiton vector 
    #


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
                    loss = margin_loss(model(X_adv_f), y[atk_filed_index],No_epoch,MT=MT,device=device).sum()
                else:
                    loss = margin_loss(model(X_adv_f), y[atk_filed_index],device=device).sum()
                tfn+=X_adv_f.shape[0]
                
        loss.backward()
        tbn+=X_adv_f.shape[0]
        if i<ADI_step_num:
            #[BEN comment] here we initialize the perturbation in the style of ODI?
            if Lnorm =="Linf":
                eta = epsilon * X_adv_f.grad.data.sign()
            elif Lnorm=="L2":
                eta = epsilon * normalize_x(X_adv_f.grad.data)
        else:
            #[BEN comment]  PGD is happening here
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
        output_loss = margin_loss(output_,y_f,device=device).squeeze()
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

def Adaptive_Auto_white_box_attack(model, device, eps, is_random, batch_size, 
                                   average_num, model_name, data_set="cifar10",
                                   Lnorm='Linf',neptune_run=None,**kwargs):
    print('Adaptive_Auto_white_box_attack',device)
    
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

    # out_restart_num = 13
    #[MODIF]
    out_restart_num = kwargs['out_restart_num']

    BIAS_ATK = False
    ADI = True
    MT=False
    wrong_logits_direct=0
    correct_logits_direct=0

    for out_re in trange(out_restart_num):
        restart_num = 1
        alpha = 1

        if out_re == 0:
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
                if out_re > 12 and kwargs['out_re_rule']:
                    max_iter += 5
                    adi_iter_num += 1
                    restart_num += 2

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
                #MODIFIED    
                if out_re > 12 and kwargs['out_re_rule']:
                    max_iter += 5
                    adi_iter_num += 1
                    restart_num += 2


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


        robust_acc = not_suc_index.sum()/len(all_atk_labs)
        neptune_run["alpha"].log(alpha)
        neptune_run["max_iter"].log(max_iter)
        neptune_run["adi_iter_num"].log(adi_iter_num)
        neptune_run["restart_num"].log(restart_num)

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

            if (out_re<10 and BIAS_ATK==False) or (out_re<11 and BIAS_ATK==True):
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
                                      MT=MT,Lnorm=Lnorm,out_re=out_re,neptune_run=neptune_run,device=device)

                    now_need_atk_index[bstart:bend] = now_need_atk_index[bstart:bend] * atk_filed_index_cold
                    now_max_loss[bstart:bend] = np.where(now_max_loss[bstart:bend]>each_max_loss,now_max_loss[bstart:bend],each_max_loss)
                    used_backward_num +=backward_num
                    total_adi_atk_suc_num+=odi_atk_suc_num
                    total_fn+=tfn

                    if out_re>=1:
                        total_odi_distribution += each_odi_distribution
                robust_acc_oneshot += now_need_atk_index[bstart:bend].sum()

            not_suc_index[not_suc_need_atk_index]=now_need_atk_index
            max_loss[not_suc_need_atk_index]= now_max_loss

            print(f'No.{len(acc_curve)} clean_acc: {clean_acc:0.4} robust acc: {not_suc_index.sum()/len(all_atk_labs):0.4} used_bp_num: {used_backward_num} '
                  f'used_fp_num: {total_fn} now_atk_num: {now_need_atk_num} now_correct_num: {robust_acc_oneshot} ')
            acc_curve.append(f"{not_suc_index.sum()/len(all_atk_labs):0.4}")

        robust_acc = not_suc_index.sum()/len(all_atk_labs)
        neptune_run["robust_acc"].log(robust_acc)

            # print(acc_curve)

#with 30 outer loop robust acc: 0.9261
#
#

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

def main(flag, model_name, ep=8./255,random=True, batch_size=128, average_number=100, device=None, neptune_run=None, **kwargs):
    # device = kwargs['device']

    stime = datetime.datetime.now()
    print(f"{flag} {model_name}")
    data_set="cifar10"
    Lnorm="Linf"

    if model_name=="TRADES":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        model = WideResNet().to(device)
        model.load_state_dict(torch.load("model_weights/TRADES/TRADES_WRN.pt"))
        ep = 0.031

    if model_name=="TRADES_mnist":
        data_set='mnist'
        from models.mnist.small_cnn import SmallCNN
        model= SmallCNN().to(device)
        model.load_state_dict(torch.load("model_weights/TRADES/model_mnist_smallcnn.pt"))
        model.to(device)
        ep=0.3
        # batch_size=512


    model.to(device)
    model.eval()
    Adaptive_Auto_white_box_attack(
        model, 
        device, 
        ep, 
        random, 
        batch_size, 
        average_number, 
        model_name, 
        data_set=data_set,
        Lnorm=Lnorm,
        neptune_run=neptune_run,
        **kwargs
    )
    etime = datetime.datetime.now()
    print(f'use time:{etime-stime}s')




def pa():
    parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack Evaluation')
    parser.add_argument('--gpuid', '-g', type=int, default=0, help='gpu to run on')
    parser.add_argument('--config', '-c', default='config/cifar10.py', help='config file location')
    args = parser.parse_args()
    return args

from mmcv import Config
import configparser
import pprint






import neptune.new as neptune


if __name__ == '__main__':


    args = pa()
    cfg = configparser.ConfigParser()
    cfg = Config.fromfile(args.config)
    print(cfg)

    pprint.pprint(cfg.model_name)
    device = torch.device("cuda:{}".format(args.gpuid))
    print(device)


    neptune_run = neptune.init(
        project="bentherien/adversarial",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQ4NTRlNi00ZGJlLTRhZDctYmRlOC0zZWM4NmE3YWE0MWIifQ==",
    )  

    neptune_run['parameters'] = cfg

    



    # cfg.kwargs.device = device

    

    # main('AAA',model_name="TRADES", average_number=500)
    # main('AAA',model_name="TRADES_mnist", average_number=1000,)
    main(flag='AAA',
         model_name=cfg.model_name,
         ep=cfg.ep,
         random=cfg.random,
         batch_size=cfg.batch_size,
         average_number=cfg.average_number,
         device=device,
         neptune_run=neptune_run,
         **cfg.kwargs
        )

    neptune_run.stop()



#flag, model_name, ep=8./255,random=True, batch_size=128, average_number=100, **kwargs):