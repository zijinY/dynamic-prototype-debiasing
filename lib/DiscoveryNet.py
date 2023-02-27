import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from lib.UNet import ResNet34Unet
from lib.modules import *
from lib.memory import DiscoveryMemory, DiscoveryMemorywithAdaptiveUpdate, DiscoveryMemorywithAdaptiveUpdate2
from lib.memory import DiscoveryMemorywithChannelAttn, DiscoveryMemoryAdaptiveUpdatewithPA, DiscoveryMemoryAdaptiveUpdatewithPA_test
from lib.memory import FixMemoryAdaptiveUpdatewithPA


class DiscoveryNet(ResNet34Unet):
    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512,
                 memory_code_size=256
                 ):
        super().__init__(num_classes=1,
                 num_channels=num_channels,
                 is_deconv=is_deconv,
                 decoder_kernel_size=decoder_kernel_size,
                 pretrained=pretrained)
        
        self.feat_channels = feat_channels
        self.memory = DiscoveryMemorywithAdaptiveUpdate(feats_size=512, code_size=memory_code_size)
        self.aux_conv = nn.Sequential(nn.Conv2d(self.feat_channels, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))                          
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
            
    def forward(self, x, epoch=None):
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Memory-based Attention ===#
        aux_out = self.aux_conv(e4)
        if self.training:
            if epoch >= 30:
                e4 = self.memory(e4, aux_out) # update 
        else:
            e4 = self.memory(e4, aux_out)
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(e4, e3, e2, e1, x)
        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)
        return aux_out, f4, f3, f2, f1

class DiscoveryNet_MS(ResNet34Unet):

    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512
                 ):
        super().__init__(num_classes=1,
                 num_channels=num_channels,
                 is_deconv=is_deconv,
                 decoder_kernel_size=decoder_kernel_size,
                 pretrained=pretrained)
        
        self.feat_channels = feat_channels
        self.memory_4 = DiscoveryMemorywithAdaptiveUpdate(feats_size=512, code_size=256)
        self.memory_3 = DiscoveryMemorywithAdaptiveUpdate(feats_size=256, code_size=128)
        self.aux_conv = nn.Sequential(nn.Conv2d(self.feat_channels, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))                          
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
            
    def forward(self, x, epoch=None):
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Memory-based Attention ===#
        aux_out = self.aux_conv(e4)
        if self.training:
            if epoch >= 20:
                e4 = self.memory_4(e4, aux_out) # update
                e3 = self.memory_3(e3, F.interpolate(aux_out, scale_factor=2, mode='bilinear', align_corners=True))
        else:
            e4 = self.memory_4(e4, aux_out)
            e3 = self.memory_3(e3, F.interpolate(aux_out, scale_factor=2, mode='bilinear', align_corners=True))
        
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(e4, e3, e2, e1, x)
        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)
        return aux_out, f4, f3, f2, f1

class DiscoveryNet_contrast(ResNet34Unet):
    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512
                 ):
        super().__init__(num_classes=1,
                 num_channels=num_channels,
                 is_deconv=is_deconv,
                 decoder_kernel_size=decoder_kernel_size,
                 pretrained=pretrained)
        
        self.feat_channels = feat_channels
        from lib.memory import DiscoveryMemorywithContrast
        self.memory = DiscoveryMemorywithContrast(feats_size=512, code_size=256)
        self.aux_conv = nn.Sequential(nn.Conv2d(self.feat_channels, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))                          
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
            
    def forward(self, x, epoch=None):
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Memory-based Attention ===#
        aux_out = self.aux_conv(e4)
        if self.training:
            if epoch >= 30:
                e4_, e4 = self.memory(e4, aux_out) # update 
            else:
                e4_ = e4
        else:
            e4_, e4 = self.memory(e4, aux_out)
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(e4, e3, e2, e1, x)

        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)
        return {"pred": [aux_out, f4, f3, f2, f1], "feat": torch.mean(e4_.flatten(start_dim=2), dim=-1)}

class DiscoveryNet_test(ResNet34Unet):
    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512,
                 memory_code_size=256
                 ):
        super().__init__(num_classes=1,
                 num_channels=num_channels,
                 is_deconv=is_deconv,
                 decoder_kernel_size=decoder_kernel_size,
                 pretrained=pretrained)
        
        self.feat_channels = feat_channels
        self.memory = DiscoveryMemoryAdaptiveUpdatewithPA_test(feats_size=512, code_size=memory_code_size)
        self.aux_conv = nn.Sequential(nn.Conv2d(self.feat_channels, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))                          
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
            
    def forward(self, x, epoch=None):
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)  
        #=== Memory-based Attention ===#
        aux_out = self.aux_conv(e4)
        if self.training:
            if epoch >= 30:
                e4 = self.memory(e4, aux_out) # update 
        else:
            feats, feats_aug = self.memory(e4, aux_out)
        return feats, feats_aug

class FixNet(ResNet34Unet):
    def __init__(self,
                memory_size,
                num_classes=1,
                num_channels=3,
                is_deconv=False,
                decoder_kernel_size=3,
                pretrained=True,
                feat_channels=512,
                memory_code_size=256
                ):
        super().__init__(num_classes=1,
                 num_channels=num_channels,
                 is_deconv=is_deconv,
                 decoder_kernel_size=decoder_kernel_size,
                 pretrained=pretrained)
        
        self.feat_channels = feat_channels
        self.memory = FixMemoryAdaptiveUpdatewithPA(memory_size=memory_size ,feats_size=512, code_size=memory_code_size)
        self.aux_conv = nn.Sequential(nn.Conv2d(self.feat_channels, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))                          
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
            
    def forward(self, x, epoch=None):
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Memory-based Attention ===#
        aux_out = self.aux_conv(e4)
        if self.training:
            if epoch >= 30:
                e4 = self.memory(e4, aux_out) # update 
        else:
            e4 = self.memory(e4, aux_out)
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(e4, e3, e2, e1, x)
        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)
        return aux_out, f4, f3, f2, f1