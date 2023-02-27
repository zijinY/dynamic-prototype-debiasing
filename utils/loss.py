import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def BCEDiceLoss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)   
    smooth = 1
    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    mask_flat = mask.view(size, -1)
    intersection = pred_flat * mask_flat
    dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + mask_flat.sum(1) + smooth)
    dice_loss = 1 - dice_score.sum()/size
    
    return (wbce + dice_loss).mean()


def SupervisedContrastiveLoss(projections, targets, temperature = 0.07):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, input, target):
        input = torch.sigmoid(input)
        N = input.size(0)
        smooth = 1
        
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = input_flat * target_flat
        
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss

        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        weit = 1 + 5*torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(input, target, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        return wbce.squeeze(1)


class ClassBalancedLoss(nn.Module):
    def __init__(self, filepath, mapping_func):
        super(ClassBalancedLoss, self).__init__()

        self.dice_criterion = DiceLoss()
        self.bce_criterion = BCELoss()
        type_mapping = mapping_func

        dist = [0 for _ in range(len(type_mapping))]
        with open(filepath) as f:
            for line in f:
                dist[type_mapping[line.split()[1]]] += 1
        num = sum(dist)
        prob = np.array([i/num for i in dist])
        # normalization
        max_prob = prob.max()
        prob = prob / max_prob
        # class reweight
        self.weight = - np.log(prob) + 1
        
    def forward(self, input, target, type):
        dice_loss = self.dice_criterion(input, target)
        bce_loss = self.bce_criterion(input, target)

        loss = (dice_loss + bce_loss) / 2
        weighted_loss = self.weighted_mean(loss, type)
        return weighted_loss
    
    def weighted_mean(self, loss, type):
        N = loss.shape[0]
        new_weight = torch.FloatTensor([self.weight[type[i].item()] for i in range(N)]).cuda()
        
        weighted_loss = torch.mean(new_weight * loss)
        return weighted_loss

def get_balancedLoss(filepath, dataset_name):
    if dataset_name == 'EndoScene':
        mapping = {'Is': 0, 'Ip': 1, 'Isp': 2, 'LST': 3}
    elif dataset_name == 'PICCOLO':
        mapping = {'IIa': 0, 'IIac': 1, 'Ip': 2, 'Is': 3, 'Isp': 4, 'unknown': 5}
    else:
        raise Exception("Invalid dataset name!")
    
    return ClassBalancedLoss(filepath, mapping)


        