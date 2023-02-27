from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.modules import *

class Memory(nn.Module):
    def __init__(self,
                 class_num=2,
                 code_size=512,
                 memory_size=20,
                 decay_rate=0.9):
    
        super(Memory, self).__init__()
        self._memory_size = memory_size
        self._code_size = code_size
        self._decay_rate = decay_rate
        self._class_num = class_num
        # memory of shape [class_num, memory_size, code_size] #
        self.memory = nn.Parameter(torch.randn(self._class_num, self._memory_size, self._code_size), requires_grad=True)
        self.label =  torch.cat([torch.tensor([0 for i in range(self._memory_size)]), torch.tensor([1 for i in range(self._memory_size)])], dim=0)

    def _get_memory(self, index=None):
        if index != None:
            return self.memory[index].data
        else:  
            return torch.cat([self.memory[i].data for i in range(self._class_num)], dim=0)
            

    def _get_label(self):
        return self.label

    def _write_to_memory(self, update_states):
        prior_memory = self.memory.data
        posterior_memory = self._decay_rate * prior_memory + (1 - self._decay_rate) * update_states
        self.memory.data = posterior_memory

    def _update(self, input):
        """
        Args:
            input : categorical encoded features of shape [class_num, batch_size, ..., height, width]
            softmax : the softmax of logit in dimension 0
        Returns:
            The update states waiting to write to the memory
        """   
        update_states = []
        for i in range(self._class_num):
            z = input[i]
            batch_size, _, = z.shape # B * C
            logit = torch.matmul(self.memory[i], z.transpose(0,1))
            attn_map = F.softmax(logit, dim=1) # S * B
            memory_update = torch.matmul(attn_map.contiguous(), z)  #S * B @ B * C = S * C
            update_states.append(memory_update.unsqueeze(0))
        update_states = torch.cat(update_states, dim=0)
        return update_states

class ClassEpisodic(nn.Module):
    def __init__(self,
                 class_num=4,
                 code_size=256,
                 memory_size=20):
    
        super(ClassEpisodic, self).__init__()
        self._memory_size = memory_size
        self._code_size = code_size
        self._class_num = class_num
        # memory of shape [class_num, memory_size, code_size] #
        self.project_head = self.X = conv2d(512, self._code_size, 1)
        self.register_buffer("queue",  torch.randn(self._class_num, self._memory_size, self._code_size))  # memory bank
        self.queue =F.normalize(self.queue, p=2, dim=2)
        self.register_buffer("queue_ptr", torch.zeros(self._class_num, dtype=torch.long))  # memory bank pointer


    @torch.no_grad()
    def _update(self, keys, preds, labels):
        
        queue = self.queue
        batch_size, feat_dim, H, W = keys.size()

        def masking(input, mask):
            batch_size, feat_dim, H, W = input.shape
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            
            return masked.view(batch_size, feat_dim, H, W)

        keys = masking(keys, preds)
        batch_size, feat_dim, H, W = keys.size()
        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_feat = torch.mean(this_feat, dim=1)
            this_label = labels[bs].contiguous().view(-1)
            ptr = int(self.queue_ptr[this_label])

            queue[this_label, ptr, :] = F.normalize(this_feat.view(-1), p=2, dim=0)
            self.queue_ptr[this_label] = (self.queue_ptr[this_label] + 1) % self._memory_size
    
    
    def _get_memory(self, label):
        return self.queue[label]

    def forward(self, feats, preds, labels):
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape
        self._update(feats, preds, labels)

        feats_augment = []
        for bs in range(batch_size):
            ori_feat = feats[bs]
            this_label = labels[bs].contiguous().view(-1)
            memory_data = self._get_memory(this_label)
            # query = S * C
            query = memory_data.squeeze(dim=0)
            # key: = C * HW
            key = feats[bs].view(dim, -1)
            # logit = S * HW (cross image relation)
            logit = query @ key
            # attn = torch.softmax(logit, 2) ##softmax维度要正确
        
            # delta = S * C
            value = memory_data.squeeze(dim=0)
            # attn_sum = C * HW
            new_feat = value.transpose(0,1) @ logit
            # new_feat = C * H * W
            new_feat = new_feat.view(-1, height, width)
            concat = torch.cat([new_feat, ori_feat], dim=0).unsqueeze(0)
            feats_augment.append(concat)

        return torch.cat(feats_augment, dim=0)

class ClassSemantic(nn.Module):
    def __init__(self,
                 class_num=4,
                 code_size=256,
                 memory_size=20,
                 decay_rate=0.9):
    
        super(ClassSemantic, self).__init__()
        self._memory_size = memory_size
        self._code_size = code_size
        self._class_num = class_num
        self._decay_rate = decay_rate
        # memory of shape [class_num, memory_size, code_size] #
        self.project_head = self.X = conv2d(512, self._code_size, 1)
        self.register_buffer("queue",  torch.randn(self._class_num, self._memory_size, self._code_size))  # memory bank
        self.queue = F.normalize(self.queue, p=2, dim=2)


    @torch.no_grad()
    def _update(self, keys, preds, labels):
        """
        Args:
            keys   : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
            labels : the paris-classification-based label of the samples 
        """
        queue = self.queue

        def masking(input, mask):
            batch_size, feat_dim, H, W = input.shape
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            
            return masked.view(batch_size, feat_dim, H, W)

        keys = masking(keys, preds)
        batch_size, feat_dim, H, W = keys.size()
        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_feat = torch.mean(this_feat, dim=1).unsqueeze(dim=1)
            this_label = labels[bs].contiguous().view(-1)
            # query = S * C
            query = queue[this_label].squeeze(dim=0)
            # key = C * 1
            key = this_feat
            # value = C * 1
            value = this_feat
            # logit = S * 1
            logit = query @ key

            update = logit @ value.transpose(0, 1)
            update = F.normalize(update.unsqueeze(0), p=2, dim=2)

            queue[this_label] = self._decay_rate * queue[this_label] + (1 - self._decay_rate) * update
        
        self.queue = queue


    def forward(self, feats, preds, labels, flag):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
            labels : the paris-classification-based label of the samples
            flag   : the indicator of the phase of the neural network (train or test)
        """ 
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape

        if flag == 'train':
            self._update(feats, preds, labels)

        feats_augment = []
        for bs in range(batch_size):
            ori_feat = feats[bs]
            this_label = labels[bs].contiguous().view(-1)
            queue = self.queue
            # query = S * C
            query = queue[this_label].squeeze(dim=0)
            # key: = C * HW
            key = feats[bs].view(dim, -1)
            # logit = S * HW (cross image relation)
            logit = query @ key
            logit = F.softmax(logit, dim=0)

            # value = S * C
            value = queue[this_label].squeeze(dim=0)
            # new_feat = C * HW
            new_feat = value.transpose(0,1) @ logit
            # new_feat = C * H * W
            new_feat = new_feat.view(-1, height, width)
            concat = torch.cat([new_feat, ori_feat], dim=0).unsqueeze(0)
            feats_augment.append(concat)

        return torch.cat(feats_augment, dim=0)

class DiscoveryMemory(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 decay_rate=0.9):
    
        super(DiscoveryMemory, self).__init__()
        self._code_size = code_size
        self._decay_rate = decay_rate
        self.project_head = conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer

        

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value, index = logit.max(dim=1)
                value = value.squeeze(0).cpu().data.numpy()
                index = index.squeeze(0).cpu().data.numpy()
                if value >= 0.5:
                    self.memory[index] = self.memory[index] * self._decay_rate + (1 - self._decay_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape
        if self.training:
            self._update_memory(feats, preds)
            print(self._get_ptr())
        

        memory = self._get_memory()
        
        
        query = memory # query = S * C
        key = feats.flatten(start_dim=2) # key: = B * C * HW
        logit = torch.matmul(query, key).transpose(0,2) # logit = HW * S * B (cross image relation)
        attn = torch.softmax(logit, 1) ##softmax维度要正确 # attn = HW * S * B
        value = memory # value = S * C
        
        feats_aug = torch.matmul(attn.transpose(1,2), value).permute(1,2,0)  # feats_aug = B * C * HW
        feats_aug = feats_aug.view(batch_size, -1, height, width) # feats_aug = B * C * H * W

        return torch.cat([feats, feats_aug], dim=1)

class DiscoveryMemorywithAdaptiveUpdate(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 ): 
    
        super(DiscoveryMemorywithAdaptiveUpdate, self).__init__()
        self._code_size = code_size
        self.project_head = conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value_i_x, index_i = logit.max(dim=1)
                value_i_x = value_i_x.squeeze(0).cpu().data.numpy()
                index_i = index_i.squeeze(0).cpu().data.numpy()
                              
                if value_i_x >= 0.5:
                    #Find the hard negative of the i#
                    logit_memory = cosine_similarity(memory, memory)
                    if ptr == 1:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,0]
                    else:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,1]
                    hard_neg_index = hard_neg_index.cpu().data.numpy()
                    
                    #Calculate the adaptive momentum
                    logit_np = logit.squeeze(0).cpu().data.numpy()
                    value_q_x = logit_np[hard_neg_index]
                    update_rate = value_q_x / (value_q_x + value_i_x)
                    self.memory[index_i] = self.memory[index_i] * update_rate + (1 - update_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape
        if self.training:
            self._update_memory(feats, preds)
            print(self._get_ptr())
        
        memory = self._get_memory()
        
        
        query = memory # query = S * C
        key = feats.flatten(start_dim=2) # key: = B * C * HW
        logit = torch.matmul(query, key).transpose(0,2) # logit = HW * S * B (cross image relation)
        attn = torch.softmax(logit, 1) ##softmax维度要正确 # attn = HW * S * B
        value = memory # value = S * C
        
        feats_aug = torch.matmul(attn.transpose(1,2), value).permute(1,2,0)  # feats_aug = B * C * HW
        feats_aug = feats_aug.view(batch_size, -1, height, width) # feats_aug = B * C * H * W

        return torch.cat([feats, feats_aug], dim=1)
        
class DiscoveryMemorywithDynamicThreshold(nn.Module):
    
    def __init__(self,
                 code_size=256,
                 decay_rate=0.9):
    
        super(DiscoveryMemorywithDynamicThreshold, self).__init__()
        self._code_size = code_size
        self._decay_rate = decay_rate
        self._threshold = None
        self.project_head = conv2d(512, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map, epoch):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        # update the threshold #
        if epoch % 10 == 0:
            self._threshold = (epoch/10 -2) * 0.4/13 + 0.3

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value, index = logit.max(dim=1)
                value = value.squeeze(0).cpu().data.numpy()
                index = index.squeeze(0).cpu().data.numpy()
                if value >= self._threshold:
                    self.memory[index] = self.memory[index] * self._decay_rate + (1 - self._decay_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds, epoch=None):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
            epoch  : the current epoch of the training procedure
        """
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape
        if self.training:
            self._update_memory(feats, preds, epoch)
            print(self._get_ptr())
        
        memory = self._get_memory()
        
        
        query = memory # query = S * C
        key = feats.flatten(start_dim=2) # key: = B * C * HW
        logit = torch.matmul(query, key).transpose(0,2) # logit = HW * S * B (cross image relation)
        attn = torch.softmax(logit, 1) ##softmax维度要正确 # attn = HW * S * B
        value = memory # value = S * C
        
        feats_aug = torch.matmul(attn.transpose(1,2), value).permute(1,2,0)  # feats_aug = B * C * HW
        feats_aug = feats_aug.view(batch_size, -1, height, width) # feats_aug = B * C * H * W
        
        

        return torch.cat([feats, feats_aug], dim=1)

class DiscoveryMemorywithAdaptiveUpdate2(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 ): 
    
        super(DiscoveryMemorywithAdaptiveUpdate2, self).__init__()
        self._code_size = code_size
        self.project_head = conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value_i_x, index_i = logit.max(dim=1)
                value_i_x = value_i_x.squeeze(0).cpu().data.numpy()
                index_i = index_i.squeeze(0).cpu().data.numpy()
                              
                if value_i_x >= 0.5:
                    #Find the hard negative of the i#
                    logit_memory = cosine_similarity(memory, memory)
                    if ptr == 1:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,0]
                    else:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,1]
                    hard_neg_index = hard_neg_index.cpu().data.numpy()
                    
                    #Calculate the adaptive momentum
                    value_i_q = logit_memory[index_i,hard_neg_index].squeeze(0).cpu().data.numpy()
                    update_rate = value_i_q / (value_i_q + value_i_x)
                    self.memory[index_i] = self.memory[index_i] * update_rate + (1 - update_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        batch_size, dim, height, width = feats.shape
        if self.training:
            self._update_memory(feats, preds)
            print(self._get_ptr())
        
        memory = self._get_memory()
        
        
        query = memory # query = S * C
        key = feats.flatten(start_dim=2) # key: = B * C * HW
        logit = torch.matmul(query, key).transpose(0,2) # logit = HW * S * B (cross image relation)
        attn = torch.softmax(logit, 1) ##softmax维度要正确 # attn = HW * S * B
        value = memory # value = S * C
        
        feats_aug = torch.matmul(attn.transpose(1,2), value).permute(1,2,0)  # feats_aug = B * C * HW
        feats_aug = feats_aug.view(batch_size, -1, height, width) # feats_aug = B * C * H * W

        return torch.cat([feats, feats_aug], dim=1)

class DiscoveryMemorywithChannelAttn(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 decay_rate=0.9):
    
        super(DiscoveryMemorywithChannelAttn, self).__init__()
        self._code_size = code_size
        self._decay_rate = decay_rate
        self.project_head = Conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer
        self.channel_attn = PolarizedChannelAttention(inplanes=self._code_size, outplanes=self._code_size)
        

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value, index = logit.max(dim=1)
                value = value.squeeze(0).cpu().data.numpy()
                index = index.squeeze(0).cpu().data.numpy()
                if value >= 0.5:
                    self.memory[index] = self.memory[index] * self._decay_rate + (1 - self._decay_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        if self.training:
            self._update_memory(feats, preds)
            print(self._get_ptr())
        
        feats_aug = self.channel_attn(feats, self._get_memory(), self._get_ptr())
        return torch.cat([feats, feats_aug], dim=1)

class DiscoveryMemoryAdaptiveUpdatewithPA(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 decay_rate=0.9):
    
        super(DiscoveryMemoryAdaptiveUpdatewithPA, self).__init__()
        self._code_size = code_size
        self._decay_rate = decay_rate
        self.project_head = Conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer
        self.channel_attn = PolarizedChannelAttention(inplanes=self._code_size, outplanes=self._code_size)

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value_i_x, index_i = logit.max(dim=1)
                value_i_x = value_i_x.squeeze(0).cpu().data.numpy()
                index_i = index_i.squeeze(0).cpu().data.numpy()
                              
                if value_i_x >= 0.5:
                    #Find the hard negative of the i#
                    logit_memory = cosine_similarity(memory, memory)
                    if ptr == 1:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,0]
                    else:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,1]
                    hard_neg_index = hard_neg_index.cpu().data.numpy()
                    
                    #Calculate the adaptive momentum
                    logit_np = logit.squeeze(0).cpu().data.numpy()
                    value_q_x = logit_np[hard_neg_index]
                    update_rate = value_q_x / (value_q_x + value_i_x)
                    self.memory[index_i] = self.memory[index_i] * update_rate + (1 - update_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        if self.training:
            self._update_memory(feats, preds)
        feats_aug = self.channel_attn(feats, self._get_memory(), self._get_ptr())
        return torch.cat([feats, feats_aug], dim=1)

class DiscoveryMemoryAdaptiveUpdatewithPA_test(nn.Module):
    def __init__(self,
                 feats_size,
                 code_size,
                 decay_rate=0.9):
    
        super().__init__()
        self._code_size = code_size
        self._decay_rate = decay_rate
        self.project_head = Conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.zeros(100, self._code_size))  # memory bank
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.int8))  # memory bank pointer
        self.channel_attn = PolarizedChannelAttention(inplanes=self._code_size, outplanes=self._code_size)

    def _get_memory(self):
        ptr = self._get_ptr()
        return self.memory[:ptr]

    def _get_ptr(self):
        ptr = self.ptr.squeeze().cpu().data.numpy()
        return ptr

    def _add_element(self, input):
        ptr = self._get_ptr()
        self.memory[ptr:ptr+1] = input
        ptr += 1
        self.ptr[0] = int(ptr)

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            ptr = self._get_ptr()
            if ptr > 0:
                memory = self._get_memory()
                logit = cosine_similarity(input[i].unsqueeze(0), memory)
                value_i_x, index_i = logit.max(dim=1)
                value_i_x = value_i_x.squeeze(0).cpu().data.numpy()
                index_i = index_i.squeeze(0).cpu().data.numpy()
                              
                if value_i_x >= 0.5:
                    #Find the hard negative of the i#
                    logit_memory = cosine_similarity(memory, memory)
                    if ptr == 1:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,0]
                    else:
                        hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,1]
                    hard_neg_index = hard_neg_index.cpu().data.numpy()
                    
                    #Calculate the adaptive momentum
                    logit_np = logit.squeeze(0).cpu().data.numpy()
                    value_q_x = logit_np[hard_neg_index]
                    update_rate = value_q_x / (value_q_x + value_i_x)
                    self.memory[index_i] = self.memory[index_i] * update_rate + (1 - update_rate) * input[i]
                else:
                    self._add_element(input[i])                                       
            elif ptr == 0:
                self._add_element(input[i])

    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        if self.training:
            self._update_memory(feats, preds)
        feats_aug = self.channel_attn(feats, self._get_memory(), self._get_ptr())
        return feats, feats_aug

class FixMemoryAdaptiveUpdatewithPA(nn.Module):
    def __init__(self,
                memory_size,
                feats_size,
                code_size,
                ): 
        super(FixMemoryAdaptiveUpdatewithPA, self).__init__()
        self._code_size = code_size
        self._memory_size = memory_size
        self.project_head = conv2d(feats_size, self._code_size, 1)
        self.register_buffer("memory",  torch.empty(self._memory_size, self._code_size))  # memory bank
        nn.init.normal_(self.memory)
        self.channel_attn = PolarizedChannelAttention(inplanes=self._code_size, outplanes=self._code_size)

    def _get_memory(self):
        return self.memory

    @torch.no_grad()
    def _update_memory(self, input, map):

        def masked_average_pooling(input, mask):
            input = input.flatten(start_dim=2)
            mask = mask.flatten(start_dim=2)
            masked = torch.mul(input, mask)  # Apply the mask using an element-wise multiply
            masked = torch.mean(masked, dim=-1)
            return masked
        
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return torch.mm(normalized_tensor_1, normalized_tensor_2.T)

        input = masked_average_pooling(input.detach(), map)
        B = input.shape[0]
        for i in range(B):
            memory = self._get_memory()
            logit = cosine_similarity(input[i].unsqueeze(0), memory)
            value_i_x, index_i = logit.max(dim=1)
            value_i_x = value_i_x.squeeze(0).cpu().data.numpy()
            index_i = index_i.squeeze(0).cpu().data.numpy()
                              
            #Find the hard negative of the i#
            logit_memory = cosine_similarity(memory, memory)
            hard_neg_index = torch.argsort(logit_memory,dim=1,descending=True)[index_i,1]
            hard_neg_index = hard_neg_index.cpu().data.numpy()
                    
            #Calculate the adaptive momentum
            logit_np = logit.squeeze(0).cpu().data.numpy()
            value_q_x = logit_np[hard_neg_index]
            update_rate = value_q_x / (value_q_x + value_i_x)
            self.memory[index_i] = self.memory[index_i] * update_rate + (1 - update_rate) * input[i]


    def forward(self, feats, preds):
        """
        Args:
            feats  : the encoded features with shape of [batch_size, feat_dim, height, width]
            preds  : the regional polyp prediction of the keys
        """
        feats = self.project_head(feats)
        if self.training:
            self._update_memory(feats, preds)
        feats_aug = self.channel_attn(feats, self._get_memory(), self._memory_size)
        return torch.cat([feats, feats_aug], dim=1)