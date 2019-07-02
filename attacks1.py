from advertorch.attacks import LinfPGDAttack
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
import torch.nn as nn
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from torchvision import transforms
from advertorch.attacks import LinfPGDAttack, L2MomentumIterativeAttack, LinfMomentumIterativeAttack, LBFGSAttack, SinglePixelAttack, LocalSearchAttack


class attack_cosine_distance(nn.CosineEmbeddingLoss):
    def __init__(self, target, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(attack_cosine_distance, self).__init__(
            margin, size_average, reduce, reduction)
        self.target = target

    def forward(self, input1, input2):
        return super(attack_cosine_distance, self).forward(input1, input2, target=self.target)


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
    learner.model.eval()
    print('learner loaded')

    if args.update:
        targets, names = prepare_facebank(
            conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    print(names)

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
    adversary = L2MomentumIterativeAttack(
        learner.model, loss_fn=attack_cosine_distance(target=torch.ones(1).to(conf.device)), eps=3,
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

    image = Image.open('/hd1/wwang/face_data/facebank/ym_t.png')

    faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)[1]
    results, score = learner.infer(conf, faces, targets, args.tta)

    ind_s = 0  # index of detected faces, 0~numb. of faces
    ind_t = 0  # index of registed template, 0~n-1, there is no "Unknown"
    print('Source face: '+names[results[ind_s] + 1])
    print('Targeted face: '+names[ind_t+1])
    adv_targeted = adversary.perturb(conf.test_transform(faces[ind_s]).to(
        conf.device).unsqueeze(0), targets[ind_t].unsqueeze(0))
    source_img = faces[ind_s]
    source_img.save('source.png')
    adv_img = transforms.ToPILImage()(transforms.Normalize(
        [-1, -1, -1], [2, 2, 2])(adv_targeted.cpu().squeeze(0)))
    adv_img.save('attack1.png')

    try:
        image = Image.open('attack1.png')
        faces = mtcnn.align_multi(
            image, conf.face_limit, conf.min_face_size)[1]
        results, score = learner.infer(conf, faces, targets, args.tta)
        for idx, _ in enumerate(faces):
            if args.score:
                print(names[results[idx] + 1]+': ', score[idx].item())
            else:
                print(names[results[idx] + 1])
    except:
        print('detect error')
