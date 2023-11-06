import torch
from torch import nn
import torch.nn.functional as F

def bn(ic):
    return nn.BatchNorm2d(ic)

def ln(ic):
    return nn.GroupNorm(ic, ic)

def relu():
    return nn.ReLU()

def prelu(ic):
    return nn.PReLU(ic)

def lrelu():
    return nn.LeakyReLU(0.2)

def selu():
    return nn.SELU()


class EqConv2d(nn.Module):
    def __init__(self, ic, oc, k, s, p, bias = False):
        super(EqConv2d, self).__init__()
        
        self.scale = nn.init.calculate_gain("conv2d") * ((ic+oc)*k*k/2)**(-0.5)
        #self.scale = nn.init.calculate_gain("conv2d") * 0.02
        self.weight = nn.Parameter(torch.randn(oc, ic, k, k))
        
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(oc))
            
        self.s = s
        self.p = p
        
    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.s, self.p)
        return x*self.scale

class EqDeconv2d(nn.Module):
    def __init__(self,ic,oc,k,s,p, bias = False):
        super(EqDeconv2d, self).__init__()
        
        self.scale = nn.init.calculate_gain("conv2d") * ((ic+oc)*k*k/2)**(-0.5)
        #self.scale = nn.init.calculate_gain("conv2d") * 0.02
        self.weight = nn.Parameter(torch.randn(ic,oc,k,k))
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(oc))
            
        self.s = s
        self.p = p
        
    def forward(self, x):
        x = F.conv_transpose2d(x, self.weight, self.bias, self.s, self.p)
        return x * self.scale

class EqLinear(nn.Module):
    def __init__(self, ic,oc):
        super(EqLinear, self).__init__()
        self.scale = nn.init.calculate_gain("linear") * (((ic+oc)/2) **(-0.5))
        #self.scale = nn.init.calculate_gain("linear") * 0.02
        self.weight = nn.Parameter(torch.randn((oc,ic)))
        self.bias = nn.Parameter(torch.zeros(oc))
    
    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x * self.scale
    
    
class MinStddev(nn.Module):
    def __init__(self):
        super(MinStddev,  self).__init__()

    def forward(self, x):
        x_std = x-x.mean(dim = 0, keepdims = True)
        x_std = torch.sqrt(x_std.pow(2) +1e-8).mean(dim=[1,2,3], keepdims = True)
        x_std = x_std * torch.ones((x.size(0), 1, x.size(2), x.size(3)), device = x.device)
        return torch.cat([x, x_std], dim=1)
      

def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
        
class ResBlk(nn.Module):
    def __init__(self, ic, oc, up):
        super().__init__()
        
        self.up=up
        
        if up:
            self.main = nn.Sequential(
                bn(ic),
                EqConv2d(ic,oc,3,1,1),
                bn(oc),
                prelu(oc),
                EqConv2d(oc,oc,3,1,1),
                bn(oc)
                
            )
            
            self.sc = nn.Sequential(
                EqConv2d(ic,oc,1,1,0),
                bn(oc)
            
            )
            
        else:
            self.main = nn.Sequential(
                ln(ic),
                EqConv2d(ic,oc,3,1,1),
                ln(oc),
                prelu(oc),
                EqConv2d(oc,oc,3,1,1),
                ln(oc)
            )
            
            self.sc = nn.Sequential(
                EqConv2d(ic,oc,1,1,0),
                ln(oc)
            
            )
    
    def forward(self, z):
        if self.up:
            z = Upscale2d(z)
        
        z_sc = self.sc(z)
        z = self.main(z)
        
        if not self.up:
            return F.avg_pool2d(z, (2,2))
        else:
            return (z_sc + z)