import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from tqdm import  tqdm,trange
import pandas as pd
import glob
import math
import random


from helpers import margin_loss, mytest_loader, normalize_x, lp_norm, mytest_loader_mem


VERBOSE = False
VALIDATE_ROB_ACC = True


def validate_rob_acc(model,x_tensor,y_tensor,adv_images_idx,adv_images_final,aaa_adv_acc,data_set='mnist'):
    print("Validating robust accuracy")
    with torch.no_grad():
        torch.cuda.empty_cache() 
        print('adv_images_idx.sum():',adv_images_idx.sum())
        # print('adv_images_idx.sum():',adv_images_idx.sum())
        missing_im_idx = torch.where(adv_images_idx == True)
        if data_set == 'mnist':
            print(adv_images_final.shape,x_tensor.shape)
            temp_x_adv = adv_images_final.clone()
            temp_x_adv[missing_im_idx] = x_tensor[missing_im_idx]
            adv_acc = (model(temp_x_adv).data.max(1)[1] == y_tensor.data).float().sum().item()

        elif data_set == 'cifar10':
            temp_x_adv = adv_images_final.clone()
            temp_x_adv[missing_im_idx] = x_tensor[missing_im_idx]
            adv_acc = 0
            for x in range(10):
                b_start, b_end = x*1000,(x+1)*1000
                X, y = Variable(temp_x_adv[b_start:b_end], requires_grad=True), Variable(y_tensor[b_start:b_end])
                out = model(X)
                correct_num = (out.data.max(1)[1] == y.data).float().sum().item()
                adv_acc += correct_num


        print('[Our] adv_acc:', adv_acc)
        print('[AAA] adv_acc:', aaa_adv_acc)
        if type(aaa_adv_acc) == torch.tensor:
            aaa_adv_acc = aaa_adv_acc.item()
        # assert(adv_acc == aaa_adv_acc)


def Adaptive_Auto_white_box_attack(model, device, eps, is_random, batch_size, 
                                   average_num, model_name,data_set="cifar10",
                                   Lnorm='Linf',neptune_run=None,np_ds=None,**kwargs):
    """Outer loop of the AAA attack which sets the parameters for each restart
    
    
    """
    torch.cuda.empty_cache()
    
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
    
    
    if np_ds == None:
        not_suc_index = np.ones(len(all_atk_labs)).astype(np.bool)
        init_atk = pd.DataFrame({'img_path':all_atk_imgs,'label':all_atk_labs,'not_suc':not_suc_index,'need_atk':not_suc_index})
        data_loader = mytest_loader(batch_size=batch_size, dataframe=init_atk,datasets=data_set,model_name=model_name)
    else:
        img_idx = [x for x in range(len(np_ds[1]))]
        label = np_ds[1]
        all_atk_labs=label
        not_suc_index = np.ones(len(label)).astype(np.bool)
        # print(len(img_idx),len(label),len(not_suc_index),len(not_suc_index))
        init_atk = pd.DataFrame({'img_idx':img_idx,'label':label,
                                 'not_suc':not_suc_index,'need_atk':not_suc_index})
        data_loader = mytest_loader_mem(np_ds=np_ds,batch_size=batch_size, dataframe=init_atk,datasets=data_set,model_name=model_name)
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

    all_x_adv = None

    x_ds, y_ds = np_ds
    if VALIDATE_ROB_ACC:
        if data_set == 'cifar10':
            x_ds = np.transpose(x_ds, (0, 3, 1, 2))
        else:
            x_ds = np.expand_dims(x_ds, axis=1)
        x_tensor = torch.from_numpy(x_ds).to(device)
        y_tensor = torch.from_numpy(y_ds).to(device)

    adv_images_final = torch.zeros(x_ds.shape,device=device,dtype=torch.float32)
    adv_images_idx = torch.tensor(not_suc_index.copy(),device=device,dtype=torch.bool)
    


    #adv_images_idx: 0 if occupied by an adversarial example, 1 otherwise

    

    #[Ben comment] This seems to be responsible for calculating the clean accuracy
    #but it also eliminates some initial advesarial examples 
    for i, test_data in enumerate(data_loader):
        bstart = i * batch_size
        bend = min((i + 1) * batch_size, len(not_suc_index))
        X, y = test_data['img'].to(device), test_data['lab'].to(device)
        # return X
        out_clean = model(X)
        total_fn+=X.shape[0]
        acc_clean_num = (out_clean.data.max(1)[1] == y.data).float().sum()
        clean_filed_index = (out_clean.data.max(1)[1] == y.data).detach().cpu().numpy()
        natural_acc += acc_clean_num
        not_suc_index[bstart:bend] = not_suc_index[bstart:bend] * clean_filed_index
        # print('clean_filed_index',clean_filed_index)
        # print('not_suc_index[bstart:bend]',not_suc_index[bstart:bend])

        adv_images_final[bstart:bend][~clean_filed_index] = X[~clean_filed_index]
        adv_images_idx[bstart:bend][~clean_filed_index] = False

        # negative_idx = np.where(~not_suc_index[bstart:bend]) #retrieve negatives from not_suc_index
        # candidates = adv_images_idx[bstart:bend][negative_idx] #retrieve negative of negative from  adv_images_idx
        # clean_idx = negative_idx[0][torch.where(candidates.cpu())] #retrieve corresponding indices
        # adv_images_final[bstart:bend][clean_idx] = X[clean_idx]  #update indices which do not already contain adv examples
        # adv_images_idx[bstart:bend][clean_idx] = False #scatter to adv image tensor


        if Lnorm =='Linf':
            random_noise = eps * torch.FloatTensor(*X.shape).uniform_(-eps, eps).sign().to(device)
            # print(random_noise.max(),random_noise.min())
            # random_noise = 
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

        adv_images_final[bstart:bend][~adv_filed_index] = X_adv[~adv_filed_index]
        adv_images_idx[bstart:bend][~adv_filed_index] = False
        # print('adv_images_idx',adv_images_idx.sum())
    
        # negative_idx = np.where(~not_suc_index[bstart:bend]) #retrieve negatives from not_suc_index
        # candidates = adv_images_idx[bstart:bend][negative_idx] #retrieve negative of negative from  adv_images_idx
        # adv_idx = negative_idx[0][torch.where(candidates.cpu())] #retrieve corresponding indices
        # adv_images_final[bstart:bend][adv_idx] = X_adv[adv_idx]  #update indices which do not already contain adv examples
        # adv_images_idx[bstart:bend][adv_idx] = False #scatter to adv image tensor


    

    max_loss[~not_suc_index]= np.ones((~not_suc_index).sum())
    clean_acc = natural_acc/len(all_atk_labs)
    print(f"clean acc:{clean_acc:0.4}")
    # out_restart_num = 13
    #[MODIF]
    out_restart_num = kwargs['out_restart_num']

    if VALIDATE_ROB_ACC:
        validate_rob_acc(model,x_tensor,y_tensor,adv_images_idx,adv_images_final,
                        aaa_adv_acc=not_suc_index.sum(),data_set=data_set)
    torch.cuda.empty_cache() 

    # exit(0)



    BIAS_ATK = False
    ADI = True
    MT=False
    wrong_logits_direct=0
    correct_logits_direct=0
    # print('[before outer loop] not_suc_index',not_suc_index.shape,not_suc_index,not_suc_index.sum())
    for out_re in trange(out_restart_num):
        restart_num = 1
        alpha = 1

        if out_re == 0:
            adi_iter_num = 0
            # max_iter = 8
            max_iter = kwargs['max_iter0']
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
                if out_re >= kwargs['orr_start_iter'] and kwargs['out_re_rule']:
                    max_iter = kwargs['orr_max_iter']
                    adi_iter_num = kwargs['orr_adi_iter_num'] 
                    restart_num = kwargs['orr_restart_num']
                    alpha = kwargs['orr_alpha']

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
                if out_re >= kwargs['orr_start_iter'] and kwargs['out_re_rule']:
                    max_iter = kwargs['orr_max_iter']
                    adi_iter_num = kwargs['orr_adi_iter_num'] 
                    restart_num = kwargs['orr_restart_num']
                    alpha = kwargs['orr_alpha']


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

        if int(not_suc_index.sum()) < len(not_suc_index)*0.1:
            alpha =1.0;restart_num +=10


        robust_acc = not_suc_index.sum()/len(all_atk_labs)
        if neptune_run != None:
            neptune_run["alpha"].log(alpha)
            neptune_run["max_iter"].log(max_iter)
            neptune_run["adi_iter_num"].log(adi_iter_num)
            neptune_run["restart_num"].log(restart_num)

        torch.cuda.empty_cache() 

        for r in range(restart_num):
            No_epoch+=1
            robust_acc_oneshot = 0

            remaining_iterations = total_iter_num - used_backward_num

            now_not_suc_num = int(not_suc_index.sum())
            need_atk_num_place1 = (len(not_suc_index)-now_not_suc_num)+min(max(100,int(now_not_suc_num*alpha)),now_not_suc_num)

            
            max_K = np.partition(max_loss, -need_atk_num_place1)[-need_atk_num_place1]
            #This line selects the indices of examples which need to be attacked 
            loss_need_atk_index = (max_loss >= max_K) & (max_loss < 0)
            not_suc_need_atk_index = loss_need_atk_index
            
            #[Ben comment] here the code sets the the indices it no longer want to attack
            init_atk['not_suc'] = not_suc_index
            #[Ben comment] Setting indices to attack now 
            init_atk['need_atk'] = not_suc_need_atk_index
            # print("[in restart_num loop] not_suc_index:",not_suc_index.sum())
            # print("[in restart_num loop] not_suc_need_atk_index:",not_suc_need_atk_index.sum())

            now_sorted_logits_index = sorted_logits_index[not_suc_need_atk_index]
            fast_init_atk = init_atk.loc[init_atk['need_atk']==1]
            now_need_atk_num = not_suc_need_atk_index.sum()
            temp_xadv = adv_images_final[not_suc_need_atk_index]#.clone()
            temp_index = adv_images_idx[not_suc_need_atk_index]#.clone()


            # print("Assert adv_images_idx.sum() == not_suc_index.sum()")
            # print(adv_images_idx.float().sum(),not_suc_index.sum())
            assert adv_images_idx.float().sum() == not_suc_index.sum()

            
            now_need_atk_index = np.ones(now_need_atk_num)
            now_max_loss = max_loss[not_suc_need_atk_index]

            if np_ds == None:
                data_loader = mytest_loader(batch_size=batch_size, dataframe=fast_init_atk, datasets=data_set, model_name=model_name)
            else:
                data_loader = mytest_loader_mem(np_ds=np_ds,batch_size=batch_size, dataframe=fast_init_atk,datasets=data_set,model_name=model_name)
            

            # print('init_atk',init_atk)
            # print('fast_init_atk',fast_init_atk)

            if (out_re<10 and BIAS_ATK==False) or (out_re<11 and BIAS_ATK==True):
                final_odi_atk = False
            else:
                final_odi_atk = True

            No_class = No_epoch
            iter_num = int(remaining_iterations/(restart_num*(out_restart_num-out_re))* now_need_atk_num)
            iter_num = min(max(iter_num,15),(max_iter+adi_iter_num))


            if out_re >= kwargs['orr_start_iter'] and kwargs['out_re_rule']:
                iter_num = kwargs['orr_max_iter'] +  adi_iter_num

            neptune_run["iter_num_in_restart"].log(iter_num)


            # print('now_need_atk_num',now_need_atk_num)
            for i, test_data in (enumerate(data_loader)):
                bstart = i * batch_size
                bend = min((i + 1) * batch_size, now_need_atk_num)
                data, target = test_data['img'].to(device), test_data['lab'].to(device)
                # print('[data]:',data.shape,'target:',target.shape)
                sorted_logits = now_sorted_logits_index[bstart:bend]
                X, y = Variable(data, requires_grad=True), Variable(target)
                if kwargs['warm_restart']:#True:
                    warm=kwargs['warm_restart_num']
                else:
                    warm=1

                for w in range(warm):
                    if w == 0:
                        X_input2 = X
                        WARM = False
                    else:
                        X_input2 = X_pgd
                        WARM = True

                        
                    atk_filed_num_cold, atk_filed_index_cold, X_pgd, each_max_loss, odi_atk_suc_num, backward_num, each_odi_distribution, tfn = \
                        AAA_white_box(model, X, X_input2, y, 
                                      eps, iter_num, adi_iter_num, WARM, 
                                      ADI, is_random, sorted_logits=sorted_logits, No_class=No_class, 
                                      w_logit_dir=wrong_logits_direct, c_logit_dir=correct_logits_direct, data_set=data_set, BIAS_ATK=BIAS_ATK,
                                      final_bias_ODI=final_odi_atk, No_epoch=No_epoch, num_class=num_class, MT=MT,
                                      Lnorm=Lnorm,out_re=out_re,neptune_run=neptune_run,device=device
                                      )

                    if VERBOSE:
                        print('atk_filed_index_cold',atk_filed_index_cold.shape,atk_filed_index_cold,atk_filed_index_cold.sum())
                        print('now_need_atk_index',now_need_atk_index.shape,now_need_atk_index,now_need_atk_index.sum())
                        print('not_suc_index',not_suc_index.shape,not_suc_index,not_suc_index.sum())
                        print('not_suc_need_atk_index',not_suc_need_atk_index.shape,not_suc_need_atk_index,not_suc_need_atk_index.sum())

                    # print("not_suc_index.sum():",not_suc_index.sum())
                    now_need_atk_index[bstart:bend] = now_need_atk_index[bstart:bend] * atk_filed_index_cold
                    # print("not_suc_index.sum():",not_suc_index.sum())
                    now_max_loss[bstart:bend] = np.where(now_max_loss[bstart:bend]>each_max_loss,now_max_loss[bstart:bend],each_max_loss)
                    used_backward_num +=backward_num
                    total_adi_atk_suc_num+=odi_atk_suc_num
                    total_fn+=tfn

                    if out_re >= 1:
                        total_odi_distribution += each_odi_distribution

                temp_xadv[bstart:bend][~atk_filed_index_cold] = X_pgd[~atk_filed_index_cold]
                temp_index[bstart:bend][~atk_filed_index_cold] = False

               
                robust_acc_oneshot += now_need_atk_index[bstart:bend].sum()
                
                

            adv_images_final[not_suc_need_atk_index] = temp_xadv
            adv_images_idx[not_suc_need_atk_index] = temp_index
            # print("[][][]not_suc_index.sum():",not_suc_index.sum())
            not_suc_index[not_suc_need_atk_index] = now_need_atk_index
            # print("[][][]not_suc_index.sum():",not_suc_index.sum())

            max_loss[not_suc_need_atk_index]= now_max_loss
            # print('[end restart_num]not_suc_index',not_suc_index.shape,not_suc_index,not_suc_index.sum())

            # print('[end restart_num]not_suc_index.sum()',not_suc_index.sum())
            # print('[end restart_num]len(all_atk_labs)',len(all_atk_labs))
            print(f'No.{len(acc_curve)} clean_acc: {clean_acc:0.4} robust acc: {not_suc_index.sum()/len(all_atk_labs):0.4} used_bp_num: {used_backward_num} '
                  f'used_fp_num: {total_fn} now_atk_num: {now_need_atk_num} now_correct_num: {robust_acc_oneshot} ')
            acc_curve.append(f"{not_suc_index.sum()/len(all_atk_labs):0.4}")
        

        robust_acc = not_suc_index.sum()/len(all_atk_labs)
        if neptune_run != None:
            neptune_run["robust_acc"].log(robust_acc)
            neptune_run["now_need_atk_num"].log(now_need_atk_num)

        if VALIDATE_ROB_ACC:
            validate_rob_acc(model,x_tensor,y_tensor,adv_images_idx,adv_images_final,
                            aaa_adv_acc=not_suc_index.sum(),data_set=data_set)
        torch.cuda.empty_cache() 

        

    
    if VALIDATE_ROB_ACC:
        pass
    else:
        if data_set == 'cifar10':
            x_ds = np.transpose(x_ds, (0, 3, 1, 2))
        else:
            x_ds = np.expand_dims(x_ds, axis=1)
        x_tensor = torch.from_numpy(x_ds).to(device)
        y_tensor = torch.from_numpy(y_ds).to(device)


    missing_im_idx = torch.where(adv_images_idx == True)
    adv_images_final[missing_im_idx] = x_tensor[missing_im_idx]

    if data_set == 'cifar10':
        adv_images_final = adv_images_final.permute(0, 2, 3, 1)
    elif data_set == 'mnist':
        adv_images_final = adv_images_final.squeeze()

    print('adv_images_final',adv_images_final.shape)
    return adv_images_final


def AAA_white_box(model, X, X_adv, y, epsilon=0.031, step_num=20, ADI_step_num=8,
                  warm = False, bias_ODI=False, random_start = True, w_logit_dir=None, c_logit_dir=None,
                  sorted_logits=None, No_class = None, data_set = None, BIAS_ATK= None,
                  final_bias_ODI = False, No_epoch = 0, num_class=10, MT=False, Lnorm='Linf', out_re=None,
                  device=None, neptune_run=None, **kwargs):
    """ Inner loop of the AAA attack which is responsible for running ODI, ADI, and scheduled PDG"""
    # print('AAA_white_box',device)
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

    if VERBOSE:
        print("At start of AAA_white_box")
        print('X_adv',X_adv.shape)
        print('X',X.shape)
        print('y',y.shape)


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
        if out_re < 3:
            raise KeyError
        p = kwargs['normal_prob']
        choice_ = np.random.choice([0,1],p=[1-p,p])

        if choice_ == 1:
            randVector_ = torch.FloatTensor(X_adv.shape[0],num_class).normal_(0.0,1.0).to(device)
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

        # print('no_r',no_r)
        # print('sorted_logits',sorted_logits.shape)

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

        if VERBOSE:
            print("In ADI and attack Loop at index ",i)
            print('X_adv_f',X_adv_f.shape)
            print('X_f',X_f.shape)
            print('y_f',y_f.shape)


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

        # print('_f_index',_f_index.shape,_f_index)
        atk_filed_index[atk_filed_index.clone()]=_f_index
        np_atk_filed_index = atk_filed_index.detach().cpu().numpy()
        if ADI_step_num > 0 and i==ADI_step_num-1:
            odi_atk_suc_num = atk_filed_index.shape[0] - atk_filed_index.float().sum()

    atk_acc_num = (model(X_adv).data.max(1)[1] == y.data).float().sum()
    tfn+=X_adv.shape[0]

    
    # print('[X_adv]',X_adv.shape)
    # print('atk_acc_num',atk_acc_num)
    # print('np_atk_filed_index',np_atk_filed_index.shape)
    # print('np_atk_filed_index',np_atk_filed_index[np.where(np_atk_filed_index == True)])
    # print('X_adv',X_adv.shape)
    # print('each_max_loss',each_max_loss)
    # print('odi_atk_suc_num',odi_atk_suc_num)
    # print('each_back_num',each_back_num)
    # print('total_odi_distribution',total_odi_distribution)
    # print('tfn',tfn)


    return atk_acc_num, np_atk_filed_index, X_adv, each_max_loss, odi_atk_suc_num, each_back_num, total_odi_distribution, tfn
