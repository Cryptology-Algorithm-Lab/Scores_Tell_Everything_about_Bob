import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import math
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from utils import utils, verification, verification_inverse
from Reconstruction import direct_attack, transfer_attack

def main(config):
    print("=========================================================")
    source_FRS = config.source
    final_target_FRS = config.target
    target_identity = config.id
    attack_type = config.type

    device = torch.device('cuda:0')
    local = utils.local_FRS().eval().to(device)
    local_inverse = utils.local_inverse().eval().to(device)
    
    _target_FRS = source_FRS
    direct_type = attack_type
    target_id = target_identity
    
    vit = False
    imgsize = 112
    if _target_FRS !='AWS':
        target_FRS = utils.target_FRS(_target_FRS, device)
        target_FRS.eval().to(device)

        if _target_FRS == 't2':
            vit = True
        elif _target_FRS == 't5':
            imgsize = 160
        else:
            vit = False
            imgsize = 112
            
        print("load the dataset to compute threshold (open-source FRS)")

        dataset_lfw = verification.load_bin('./utils/dataset/lfw.bin', (imgsize, imgsize))
        th = verification.test(dataset_lfw, target_FRS, 64, device, 10, vit)
        th_angle = math.acos((th-2)/(-2))*180/math.pi
    else:
        target_FRS='AWS'
        th_angle = 80
        
    args = edict()
    args.device = device
    args.local = local
    args.local_inverse = local_inverse
    ofs = utils.ofs_load()
    ofs = ofs.to(device)
    args.ofs = ofs
    args.bb_FRS = target_FRS
    args.bb_th = th_angle
    args.bb_imgsize = imgsize
    args.direct_type = direct_type
    args.target_id = target_id
    args.vit = vit
    
    if source_FRS=='AWS' and final_target_FRS!='AWS':
        pass
    else:
        recon, cos = direct_attack(args)

    targetimg1 = utils.targetimg_type1(imgsize=args.bb_imgsize, target_id=args.target_id)

    if direct_type==2:
        targetimg2 = utils.targetimg_type2(imgsize=args.bb_imgsize, target_id=args.target_id)
    
    if source_FRS == final_target_FRS:
    
        if target_FRS!='AWS':
            print(" ")
            print('Angle distance:', math.acos(cos)*180/math.pi)
            print('Threshold:', args.bb_th)
            if math.acos(cos)*180/math.pi<=args.bb_th:
                print("success!!")
            else:
                print("fail..")
        else:
            print(" ")
            print('Confidence score:', cos)
            print('Threshold:', args.bb_th)
            if cos>=args.bb_th:
                print("success!!")
            else:
                print("fail..")
            
    else:

        _target_FRS_transfer = final_target_FRS
        transfer_type = attack_type
        target_id = target_identity

        transfer_vit = False
        imgsize_transfer = 112

        if _target_FRS_transfer !='AWS':
            target_FRS_transfer = utils.target_FRS(target_FRS = _target_FRS_transfer).eval().to(device)

            if _target_FRS_transfer == 't2':
                transfer_vit = True
                print(" ")
                dataset_lfw = verification.load_bin('./utils/dataset/lfw.bin', (imgsize_transfer,imgsize_transfer))
            elif _target_FRS_transfer == 't5':
                imgsize_transfer = 160
                print(" ")
                dataset_lfw = verification.load_bin('./utils/dataset/lfw.bin', (imgsize_transfer,imgsize_transfer))
            else:
                transfer_vit = False
                print(" ")
                dataset_lfw = verification.load_bin('./utils/dataset/lfw.bin', (imgsize_transfer,imgsize_transfer))

            th_transfer = verification.test(dataset_lfw, target_FRS_transfer, 64, device, 10, transfer_vit)
            th_transfer_angle = math.acos((th_transfer-2)/(-2))*180/math.pi
            direct_type = None
        else:
            target_FRS_transfer='AWS'
            th_transfer_angle = 80

        args = edict()
        args.device = device
        args.local = local
        args.local_inverse = local_inverse
        args.ofs = utils.ofs_load()
        args.bb_FRS = target_FRS
        args.bb_th = th_angle
        args.bb_imgsize = imgsize
        args.direct_type = direct_type
        args.transfer_FRS = target_FRS_transfer
        args.transfer_th = th_transfer_angle
        args.transfer_imgsize = imgsize_transfer
        args.transfer_type = transfer_type
        args.target_id = target_id
        args.vit = vit
        args.transfer_vit = transfer_vit

        recon, cos = transfer_attack(args)

        
        targetimg1 = utils.targetimg_type1(imgsize=args.bb_imgsize, target_id=args.target_id)
       
        if transfer_type==2:
            targetimg2 = utils.targetimg_type2(imgsize=args.bb_imgsize, target_id=args.target_id)
            
        if target_FRS_transfer!='AWS':
            print(" ")
            print('Angle distance:', math.acos(cos)*180/math.pi)
            print('Threshold:', args.transfer_th)
            if math.acos(cos)*180/math.pi<=args.transfer_th:
                print("success!!")
            else:
                print("fail..")
        else:
            print(" ")
            print('Confidence score:', cos)
            print('Threshold:', args.transfer_th)
            if cos>=args.transfer_th:
                print("success!!")
            else:
                print("fail..")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--source", type=str, default='t1', help="source FRS")
    parser.add_argument("--target", type=str, default='t1', help="target FRS")
    parser.add_argument("--id", type=int, default=310, help="target identity")
    parser.add_argument("--type", type=int, default=1, help="attack type, same image (1) or diffenrent image (2) from same identity")
    main(parser.parse_args())