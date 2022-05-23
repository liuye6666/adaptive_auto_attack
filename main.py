""" Launcher file for modified A^3 attack from (https://github.com/liuye6666/adaptive_auto_attack)
for an assignment

All credits for the original method got to Liu et al. https://arxiv.org/abs/2203.05154


Modifier: Benjamin Therien


"""
import configparser
import pprint
import argparse
import torch
import datetime
import os
import sys

import numpy as np
import os.path as osp
import neptune.new as neptune

from aaa import Adaptive_Auto_white_box_attack
from mmcv import Config




# os.environ['CUDA_VISIBLE_DEVICES'] = '6'



def main(flag, model_name, ep=8./255,random=True, batch_size=128, average_number=100, device=None, neptune_run=None, **kwargs):

    stime = datetime.datetime.now()
    print(f"{flag} {model_name}")
    data_set="cifar10"
    Lnorm="Linf"

    if model_name=="TRADES":
        from models.CIFAR10.TRADES_WRN import WideResNet # TRADES_WRN
        model = WideResNet().to(device)
        model.load_state_dict(torch.load("model_weights/TRADES/model_cifar_wrn.pt"))
        # ep = 0.031

    if model_name=="TRADES_mnist":
        data_set='mnist'
        from models.mnist.small_cnn import SmallCNN
        model= SmallCNN().to(device)
        model.load_state_dict(torch.load("model_weights/TRADES/model_mnist_smallcnn.pt"))
        model.to(device)
        # ep=0.3
        # batch_size=512


    model.to(device)
    model.eval()
    adversarial_images = Adaptive_Auto_white_box_attack(
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

    return adversarial_images




def pa():
    parser = argparse.ArgumentParser(description='PyTorch MNIST A3 Attack Evaluation')
    parser.add_argument('--gpuid', '-g', type=int, default=0, help='gpu to run on')
    parser.add_argument('--config', '-c', default='config/mnist.py', help='config file location')
    parser.add_argument('--save-path', '-sp', default='mnist_adv/', help='where to save the adv examples')
    parser.add_argument('--no-neptune', '-nn', action='store_true', help='do not use neptune')
    parser.add_argument('--trades-path', '-t', default='/home/therien/Documents/github/TRADES', help='path to trades repo')
    args = parser.parse_args()
    return args

#python main.py -g 0 -c config/mnist.py -sp mnist_adv -t /home/therien/Documents/github/TRADES
#python main.py -g 0 -c config/cifar10.py -sp cifar10_adv -t /home/therien/Documents/github/TRADES

if __name__ == '__main__':


    args = pa()
    cfg = configparser.ConfigParser()
    cfg = Config.fromfile(args.config)
    print(cfg)

    pprint.pprint(cfg.model_name)
    device = torch.device("cuda:{}".format(args.gpuid))
    print(device)


    if args.no_neptune:
        neptune_run = None
    else:
        neptune_run = neptune.init(
            project="bentherien/adversarial",
            api_token="",
        )  
        neptune_run['parameters'] = cfg

    trades_path = args.trades_path
    print("loading {}".format(osp.join(trades_path,f'data_attack/{cfg.dataset}_X.npy')))
    print("loading {}".format(osp.join(trades_path,f'data_attack/{cfg.dataset}_Y.npy')))
    trades_X = np.load(osp.join(trades_path,f'data_attack/{cfg.dataset}_X.npy')).astype(np.float32)
    trades_y = np.load(osp.join(trades_path,f'data_attack/{cfg.dataset}_Y.npy')).astype(np.int64)
    print('trades_X',trades_X.shape)
    print('trades_y',trades_y.shape)
    
    adversarial_images = main(flag='AAA',
         model_name=cfg.model_name,
         ep=cfg.ep,
         random=cfg.random,
         batch_size=cfg.batch_size,
         average_number=cfg.average_number,
         device=device,
         neptune_run=neptune_run,
         np_ds=(trades_X,trades_y,),
         **cfg.kwargs
        )

    torch.cuda.empty_cache() 

    f_list = os.listdir(args.save_path)
    nums = [int(x[len(cfg.dataset):-4]) for x in f_list if x[-4:] == '.npy']
    if nums == []:
        filename = "{}0.npy".format(cfg.dataset)
    else:
        filename = "{}{}.npy".format(cfg.dataset,np.max(nums) + 1)


    neptune_run['images_filename'] = filename
    


    print('Saving adversarial images of shape:{} to {}'.format(adversarial_images.shape,osp.join(args.save_path,filename)))
    save_trades_pt = osp.join(trades_path,'data_attack',f'{cfg.dataset}_X_adv.pt')
    torch.save(
        adversarial_images.detach().cpu(),
        open(save_trades_pt,'wb')
    )

    torch.save(
        adversarial_images.detach().cpu(),
        open(osp.join(args.save_path,filename[:-4]+".pt"),'wb')
    )

    np.save(
        file=osp.join(args.save_path,filename), 
        arr=adversarial_images.detach().cpu().numpy(), 
        allow_pickle=True
    )
    



    # ============ Testing with the TRADES repo ============



    if not osp.isdir(osp.join(trades_path,'data_attack')):
        os.mkdir(osp.join(trades_path,'data_attack'))




    def shape_perturbation(X, X_adv, eps=cfg.ep, extra=.0):
        """Method for Clamping adversarial perturbations  
        
        args:
            X (np.ndarray): The clean images
            X_adv (torch.tensor): The adversarial images 
        """
        eps += extra
        print("eps [{},{}]".format(-eps, eps))
        X_ = torch.from_numpy(X.copy()) 
        X_adv_ = X_adv.detach().cpu().clone()
        print("eps [{},{}]".format(-eps, eps))
        eta = X_adv_ - X_.clone()

        print('[Initial]     eta:{},min:{},max:{}'.format(eta.shape,eta.min(),eta.max()))
        
        eta = torch.clamp(eta,-eps, eps)

        print('[After clamp] eta:{},min:{},max:{}'.format(eta.shape,eta.min(),eta.max()))
        final_X_adv = (eta + X_).numpy()#adversarial_images.detach().cpu().numpy()

        preturbation_after = final_X_adv.copy() - X.copy()
        print('[After NP]    eta:{},min:{},max:{}'.format(preturbation_after.shape,preturbation_after.min(),preturbation_after.max()))

        return final_X_adv

    # eps - 0.00000004 is the closest we get to actual adversarial images
    # adversarial_images = shape_perturbation(trades_X, adversarial_images, cfg.ep, extra=-0.00000004)

    save_trades = osp.join(trades_path,'data_attack',f'{cfg.dataset}_X_adv.npy')
    print('Saving adversarial images of shape:{} to {}'.format(adversarial_images.shape,save_trades))
    np.save(
        file=save_trades, 
        arr=adversarial_images.detach().cpu().numpy(), 
        allow_pickle=True
    )
    
    model_filename = 'model_mnist_smallcnn.pt' if cfg.dataset == 'mnist' else 'model_cifar_wrn.pt'

    ev = f'evaluate_attack_{cfg.dataset}.py' 
    ev = osp.join(trades_path,ev)
    command = "{} {} --data-attak-path {} --model-path {} --data-path {} --target-path {}".format(
            sys.executable,ev,save_trades, osp.join(trades_path,'checkpoints',model_filename), 
            osp.join(trades_path,f'data_attack/{cfg.dataset}_X.npy'),osp.join(trades_path,f'data_attack/{cfg.dataset}_Y.npy')
        )




    if not args.no_neptune:
        neptune_run.stop()

    print('Executing command: {}'.format(command))
    os.system(command)


    exit(0)

