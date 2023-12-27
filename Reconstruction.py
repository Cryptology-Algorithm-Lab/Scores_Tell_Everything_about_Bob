import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import math
import numpy as np
from PIL import Image, ImageFile
from tqdm.auto import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import os

from facenet_pytorch import *
from onnx2torch import convert

from utils.utils import get_log, img_load, img2ten, img2ten_vit, targetimg_type1, targetimg_type2, blackbox, local_FRS, local_inverse
from utils.AWS import cosine_score_AWS, query_AWS

def direct_attack(args):
    if os.path.exists('results'):
        assert os.path.isdir('results'), 'it exists but is not a folder'
    else:
        os.makedirs('results')    

    savepath = './results/direct_type1_'+str(args.target_id)+'.png'
    targetimg = targetimg_type1(imgsize=args.bb_imgsize, target_id=args.target_id)
    ofs = args.ofs
    # Need to resize the OFS for the blackbox...
    if args.bb_imgsize!=112:
        ofsb = F.interpolate(ofs, (args.bb_imgsize,args.bb_imgsize), mode='nearest').to(args.device)
    else:
        ofsb = ofs.to(args.device)
    
    with torch.no_grad():
        # local process
        # ofs : N*3*112*112 image tensor of orthogonal face set
        # query_local : N * D  matrix (before normalize)
        # A : N * D matrix (after normalization, N feature vector)
        query_local = args.local(ofs)            
        A = F.normalize(query_local)
        
        # Query to blackbox
        # coss : N cosine similarity between target and ofs
        if args.bb_FRS!='AWS':
            if args.vit == True:
                decisions, coss = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten_vit(targetimg).to(args.device), (ofs*255+127.5).to(args.device), args.device) 
            else:
                decisions, coss = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten(targetimg).to(args.device), ofsb, args.device)
        else:
            coss = cosine_score_AWS(args.target_id).to(args.device)

        # Reconstruct
        # x_hat : approximated feature vector of target image using pseudo inverse of A and cosine similairty
        # recon : output image tensor of x_hat (size : 128*128, range : [-1,1], before interpolation)
        # recon : output image tensor of x_hat (size : 112*112 or 160*160, range : [-1,1], after interpolation)
        x_hat = torch.mm(torch.linalg.pinv(A),coss.reshape(-1,1))
        recon = args.local_inverse(F.normalize(x_hat.T))
        recon = F.interpolate(recon.to(args.device), (args.bb_imgsize, args.bb_imgsize), mode = 'nearest')
        
        # img_prime : image tensor (range : [0,1])
        img_prime =((recon+1)/2).detach().cpu().squeeze(0).numpy().transpose(1,2,0)
        plt.imsave(savepath, img_prime)

        img_prime = Image.open(savepath)
        img_prime = img_load(img_prime, args.bb_imgsize)

        # Query to blackbox to impersonate
        # vit is Face Transformer based on Vision Transformer whose pre-processing is different from other face recognition model
        if args.bb_FRS!='AWS':
            if args.vit == True:
                d, cos = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten_vit(targetimg).to(args.device), img2ten_vit(img_prime).to(args.device), args.device)
            else:
                d, cos = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten(targetimg).to(args.device), img2ten(img_prime).to(args.device), args.device)
                
            # If something in the OFS has a better similarity score than our reconstruction, then we use the best image from the OFS
            # Note that this special case was handled slightly differently for our paper's experiments
            # where we picked arbitrarily from the OFS an image that crossed the threshold for a given system
            # We updated it here for better consistency, so a few reconstructed images for t4, t5 will be changed
            # This leads to slightly better attack success rates for those FRSs.
            if (coss.max() > cos):
                # print('special case',cos,coss.max(),coss.argmax(),coss[coss.argmax()],(decisions==1).sum())
                img_prime_tmp = ofs[coss.argmax()]
                img_prime_tmp = ((img_prime_tmp+1)/2).detach().cpu().numpy().transpose(1,2,0)      
                cos = coss.max()
                plt.imsave(savepath, img_prime_tmp)
                img_prime = Image.open(savepath)
                img_prime = img_load(img_prime, args.bb_imgsize)

            # attack_type = 2 means target is another image from same target identity. (harder than type 1)
            if args.direct_type==2:
                targetimg = targetimg_type2(imgsize=args.bb_imgsize, target_id=args.target_id) 

                # Query to blackbox
                if args.vit == True:            
                    d, cos = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten_vit(targetimg).to(args.device), img2ten_vit(img_prime).to(args.device), args.device)
                else:
                    d, cos = blackbox(args.bb_FRS, args.bb_th, args.bb_imgsize, args.bb_imgsize, img2ten(targetimg).to(args.device), img2ten(img_prime).to(args.device), args.device)    
            return img_prime, cos
        
        else:
            confidence = query_AWS(args.target_id, savepath, args.direct_type)                    
            return img_prime, confidence      


def transfer_attack(args):
    if os.path.exists('results'):
        assert os.path.isdir('results'), 'it exists but is not a folder'
    else:
        os.makedirs('results')           
    
    targetimg = targetimg_type1(imgsize=args.bb_imgsize, target_id=args.target_id) 
    with torch.no_grad():
        # Reconstruct and save the target image using direct_attack
        recon, cos_direct = direct_attack(args)
        
        # Query to blackbox to impersonate
        if args.transfer_FRS!='AWS':
            if args.transfer_vit == True:         
                d, cos = blackbox(args.transfer_FRS, args.transfer_th, args.bb_imgsize, args.transfer_imgsize, img2ten_vit(targetimg).to(args.device), img2ten_vit(recon).to(args.device), args.device)
            else:        
                d, cos = blackbox(args.transfer_FRS, args.transfer_th, args.bb_imgsize, args.transfer_imgsize, img2ten(targetimg).to(args.device), img2ten(recon).to(args.device), args.device)

            if args.transfer_type==2:
                targetimg = targetimg_type2(imgsize=args.bb_imgsize, target_id=args.target_id)

                # Query to blackbox
                if args.transfer_vit == True:                                
                    d, cos = blackbox(args.transfer_FRS, args.transfer_th, args.bb_imgsize, args.transfer_imgsize, img2ten_vit(targetimg).to(args.device), img2ten_vit(recon).to(args.device), args.device)
                    #print(cos)
                else:                               
                    d, cos = blackbox(args.transfer_FRS, args.transfer_th, args.bb_imgsize, args.transfer_imgsize, img2ten(targetimg).to(args.device), img2ten(recon).to(args.device), args.device)

            return recon, cos
        
        else:
            savepath = './results/direct_type1_'+str(args.target_id)+'.png'
            confidence = query_AWS(args.target_id, savepath, args.transfer_type)                    
            return recon, confidence      

