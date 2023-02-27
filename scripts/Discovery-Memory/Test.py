import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from tqdm import tqdm
import sys
sys.path.append(".")
from lib.DiscoveryNet import DiscoveryNet, FixNet
from utils.dataloader import test_dataset
from utils.eval import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/FixMemoryAdaptiveUpdatewithPA/30/{}/model-150.pth')
parser.add_argument('--data_root', type=str, default='/data/yinzijin/Polyp_raw/{}/TestDataset/')
parser.add_argument('--save_root', type=str, default='/data/yinzijin/Polyp_pred/FixMemoryAdaptiveUpdatewithPA/30/{}/')
parser.add_argument('--gpu', type=str, default='1')

for _data_name in ['PICCOLO']:
    save = True
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    data_path = opt.data_root.format(_data_name)
    save_path = opt.save_root.format(_data_name)
    load_path = opt.pth_path.format(_data_name)
    
    model = FixNet(memory_size=30)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(load_path))
    model = model.cuda()
    model.eval()
    if save == True:
        os.makedirs(save_path, exist_ok=True)
    
    test_loader = test_dataset(root=data_path, testsize=opt.testsize)
    evaluator = Evaluator()
    
    bar = tqdm(test_loader.images)
    #print(model.module.memory._get_ptr())
    for i in bar:
        image, gt, name = test_loader.load_data()
        image = image.cuda()
        gt = gt.cuda()
        output = model(image)
        res = output[len(output)-1]
        res = F.upsample(res, size=(gt.shape[2], gt.shape[3]), mode='bilinear', align_corners=False)[0,0]
        res[torch.where(res>0)] /= (res>0).float().mean()
        res[torch.where(res<0)] /= (res<0).float().mean()
        res = torch.sigmoid(res)
        evaluator.update(res, gt)

        if save == True:
            res = res.data.cpu().numpy().squeeze()
            misc.imsave(save_path+name, res)
    evaluator.show()