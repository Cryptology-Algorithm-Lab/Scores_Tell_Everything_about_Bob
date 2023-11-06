import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from utils.utils import local_FRS, local_inverse


def GenOFS(device, dataset, ofs_num=99, savepath='./param/ofs.pt'):
    print('# dataset:', len(dataset))
    
    local = local_FRS(device)
    local = local.eval().to(device)
    nbnet = local_inverse(device)  
    nbnet = nbnet.eval().to(device)    
    
    img = dataset.__getitem__(0)[0]
    temp_init = local(img.unsqueeze(0).to(device))
    recon = nbnet(F.normalize(temp_init))
    recon = F.interpolate(recon, (112,112), mode = 'nearest')
    temp = local(recon)

    ofs = torch.zeros(ofs_num,3,112,112)
    ofs[0] = recon

    j=1
    with torch.no_grad():
        for id in range(len(dataset)):
            if id%1000 == 0:
                print('id = ', id)
            img = dataset.__getitem__(id)[0]
            temp_init = local(img.unsqueeze(0).to(device))
            temp_init = F.normalize(temp_init)
            basis2 = nbnet(temp_init)
            basis2 = F.interpolate(basis2, (112,112), mode = 'nearest')
            temp2 = local(basis2)

            cos = F.cosine_similarity(temp, temp2)
            if (torch.abs(cos)<0.0871).all():  
                temp = torch.cat([temp,temp2])
                ofs[j] = basis2.detach().cpu()
                print('---------------------------------num:', len(temp))
                j+=1 
            if j==ofs_num+1:
                print('Done.')
                break

    torch.save(ofs, savepath)
    print('Saved.', savepath)
    
     
    
def orthogonal_check(ofs, model):
    print('# OFS:', ofs.size(0))

    total_min=[]
    total_max=[]
    with torch.no_grad():
        for i in range(ofs.size(0)):
            temp1 = model(ofs[i].unsqueeze(0))
            local_cos=[]
            for j in range(ofs.size(0)):
                temp2 = model(ofs[j].unsqueeze(0))
                cos = F.cosine_similarity(temp1, temp2)
                if i!=j:
                    local_cos.append(cos)

            max_angle = math.acos(min(local_cos))*180/math.pi
            min_angle = math.acos(max(local_cos))*180/math.pi
            print('i={} Min angle:{}, Max angle:{}'.format(i, min_angle, max_angle))
            
            total_min.append(min_angle)
            total_max.append(max_angle)            
            
    print('[RESULT] Min angle:{}, Max angle:{}'.format(min(total_min), max(total_max)))
    
    