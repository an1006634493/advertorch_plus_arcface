# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:17:52 2019

@author: kang
"""

from advertorch.attacks import LinfPGDAttack
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
import model
import torch.nn as nn
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
from advertorch.attacks import LinfPGDAttack, L2MomentumIterativeAttack, LinfMomentumIterativeAttack, LBFGSAttack, SinglePixelAttack, LocalSearchAttack
import torchsnooper as ts
from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from tqdm import tqdm
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm

class attack_arc(nn.CrossEntropyLoss):
    def _init_(self):
        super(attack_arc, self)._init_()
    
    def forward(self, embeddings, labels):
        thetas = head(embeddings, labels)
        loss = super(attack_arc, self).forward(thetas, labels)
        return loss

def get_classnum(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return class_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",
                        action="store_true")
    parser.add_argument(
        '-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument(
        "-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument(
        "-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument(
        "-c", "--score", help="whether show the confidence score", action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_ir_se50.pth', False, True)
    else:
        learner.load_state(conf, 'ir_se50.pth', False, True)
        #learner.load_state(conf, 'final.pth', False, True)
    learner.model.eval()
    learner.model = learner.model.to(conf.device)
    print('learner loaded')

    torch.manual_seed(0)

    '''
    # targeted L-BFGS attack
    adversary = LBFGSAttack(
        learner.model, loss_fn=attack_cosine_distance(target=torch.ones(1).to(conf.device)), num_classes=1,
        clip_min=-1.0, clip_max=1.0, targeted=False)
    '''
    '''
    #targeted Linf attack    
    adversary = LinfPGDAttack(
        learner.model, loss_fn=attack_cosine_distance(target=torch.ones(1).to(conf.device)), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0, clip_max=1.0,
        targeted=True)
    '''
    
    #targeted L2 attack    

    #print(target_test)
    #adversary = L2MomentumIterativeAttack(
    #    learner.model, loss_fn=attack_cosine_distance(target=target_test), eps=3,
    #    nb_iter=40, eps_iter=0.1, decay_factor=1., clip_min=-1.0, clip_max=1.0,
    #    targeted=True)
    class_num = get_classnum(conf.emore_folder/'imgs')
    adversary = L2MomentumIterativeAttack(
        learner.model, loss_fn=attack_arc(), eps=3,
        nb_iter=40, eps_iter=0.1, decay_factor=1., clip_min=-1.0, clip_max=1.0,
        targeted=True)


    '''
    # no-targeted L2 attack through self-targeted
    adversary = L2MomentumIterativeAttack(
        learner.model, loss_fn=attack_cosine_distance(target=-torch.ones(1).to(conf.device), margin=0.1), eps=10,
        nb_iter=40, eps_iter=0.2, decay_factor=1., clip_min=-1.0, clip_max=1.0,
        targeted=True)
    '''

    '''
    #no-targeted L2 attack    
    adversary = L2MomentumIterativeAttack(
        learner.model, loss_fn=attack_cosine_distance(target=torch.ones(1).to(conf.device)), eps=10,
        nb_iter=40, eps_iter=0.2, decay_factor=1., clip_min=-1.0, clip_max=1.0,
        targeted=False)
    '''

    #image = Image.open("/hd1/anshengnan/InsightFace_Pytorch/data/test.jpg")
    loader, class_num = get_train_loader(conf)
    
    head = Arcface(embedding_size=conf.embedding_size, classnum=class_num).to(conf.device)
    
    with ts.snoop():
      for imgs, labels in tqdm(iter(loader)):
          imgs = imgs.to(conf.device)
          labels = labels.to(conf.device)
          embeddings = learner.model(imgs)
          thetas = head(embeddings, labels)
          adv_targeted = adversary.perturb(imgs, labels)
          adv_embeddings = learner.model(adv_targeted)
          adv_thetas = head(adv_embeddings, labels)
          print(labels)
          thetas = list(thetas.squeeze(0))
          adv_thetas = list(adv_thetas.squeeze(0))
          print(thetas.index(max(thetas)))
          print(adv_thetas.index(max(adv_thetas)))
          break
