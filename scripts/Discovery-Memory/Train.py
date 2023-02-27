import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import numpy as np
import random
from tensorboardX import SummaryWriter
import sys
sys.path.append(".")
from utils.loss import BCEDiceLoss
from utils.dataloader import get_loader, get_naive_loader
from utils.utils import AvgMeter
from lib.DiscoveryNet import DiscoveryNet, FixNet

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def train(train_loader, model, optimizer, epoch, writer):
    model.train()
    loss_total_record = AvgMeter()
    loss_record = AvgMeter()
    loss_aux_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):           
        optimizer.zero_grad()
        # ---- data prepare ----
        data = pack
        images, gts = data['image'], data['label']
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # ---- forward ----
        pred = model(images, epoch)
        # ==== loss function ====
        loss4 = BCEDiceLoss(pred[4], gts)
        loss3 = BCEDiceLoss(pred[3], gts)
        loss2 = BCEDiceLoss(pred[2], gts)
        loss1 = BCEDiceLoss(pred[1], gts)
        loss = loss1 + loss2 + loss3 + loss4
        # ---- auxiliary loss function ----
        loss_aux = BCEDiceLoss(pred[0], gts)
        loss_total = loss_aux + loss
        # ---- backward ----
        loss_total.backward()
        optimizer.step()
        # ---- recording loss ----
        loss_record.update(loss.data, opt.batchsize)
        loss_aux_record.update(loss_aux.data, opt.batchsize)
        loss_total_record.update(loss_total.data, opt.batchsize)

        writer.add_scalar('Train/Total Loss', loss_total.data, epoch * len(train_loader) + i)
        writer.add_scalar('Train/Auxiliary Loss', loss_aux.data, epoch * len(train_loader) + i)

        # ---- train visualization ----
        if i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[total loss: {:.4f}, aux loss: {:.4f}, loss: {:.4f}] '. 
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_total_record.show(), loss_aux_record.show(), loss_record.show()))
    # ---- recording memory states ----
    ptr = model.module.memory._get_ptr()
    if ptr >= 1:
        memory = model.module.memory._get_memory().cpu().data.numpy()
        np.savez_compressed(os.path.join(opt.memory_record_path,str(epoch)+'.npz'),memory=memory)

    # ---- saving the model ----
    #save_path = './snapshots/{}/'.format(opt.save_root)
    #os.makedirs(save_path, exist_ok=True)
    #if epoch % 50 == 0:
    #    torch.save(model.state_dict(), save_path + 'model-%d.pth' % epoch)
    #    print('[Saving Snapshot:]', save_path + 'model-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='/data/yinzijin/Polyp_raw/PICCOLO/TrainDataset/', help='path to train dataset')
    parser.add_argument('--save_root', type=str,
                        default='FixMemoryAdaptiveUpdatewithPA/30/PICCOLO')
    parser.add_argument('--memory_record_path', type=str,
                        default='./memory_records/DiscoveryMemorywithAdaptiveUpdate2/')
    parser.add_argument('--gpu', type=str,
                        default='0', help='used GPUs')
    opt = parser.parse_args()

    # ---- build models ----
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    model = DiscoveryNet()
    model = nn.DataParallel(model).cuda()
    writer = SummaryWriter()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    
    train_loader = get_naive_loader(root=opt.train_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    os.makedirs(opt.memory_record_path, exist_ok=True)

    for epoch in range(1, opt.epoch+1):
        train(train_loader, model, optimizer, epoch, writer)


