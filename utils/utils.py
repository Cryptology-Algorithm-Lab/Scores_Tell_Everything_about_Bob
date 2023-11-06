from PIL import Image, ImageFile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import mxnet as mx
from datetime import datetime
import numbers
import os
import queue as Queue
import threading
from typing import Iterable

from .iresnet import iresnet50, iresnet100
from .nbnet import NbNet

from vit_pytorch import ViTs_face
from onnx2torch import convert
from .sfnet_deprecated import *
from facenet_pytorch import InceptionResnetV1



def get_log(txt, filename):
    date = datetime.now().strftime('%Y%m%d %H:%M:%S')
    f = open('./log_{}.txt'.format(filename), 'a')
    f.write("[{}] ".format(date) + txt + "\n")
    f.close()      
    return


def get_dataset(root_dir, local_rank = 0, batch_size = 128):
    dataset = MXFaceDataset(root_dir, local_rank)
    return dataset
   
    
class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((112,112)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label) #, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


def img_load(img, imgsize=112):  
    img = img.resize((imgsize,imgsize))
    img = img.convert('RGB')
    return np.array(img)


def img2ten(img_array):
    img_tensor = (torch.Tensor((img_array-127.5)/255)).permute(2,0,1).unsqueeze(0)
    return img_tensor

def img2ten_vit(img_array):
    img_tensor = (torch.Tensor(img_array)).permute(2,0,1).unsqueeze(0)
    return img_tensor

def ofs_load():
    ofs = torch.zeros(99,3,112,112)
    for i in range(99):
        path = './sample_OFS/'+str(i)+'.png'
        img = Image.open(path)
        ofs[i] = img2ten(img_load(img, 112))
    return ofs


def targetimg_type1(imgsize=112, target_id=310):
    path = './sample_LFW/L/'+str(target_id)+'.png'
    img = Image.open(path)
    img1 = img_load(img, imgsize)
    return img1


def targetimg_type2(imgsize=112, target_id=310):
    path = './sample_LFW/R/'+str(target_id)+'.png'
    img = Image.open(path)
    img2 = img_load(img, imgsize)
    return img2


def local_FRS(device='cpu'):
    model = iresnet50(pretrained=False, progress=True)
    model.load_state_dict(torch.load('./utils/param/local.pt', map_location = device))
    model = model.eval().to(device)
    return model            


def local_inverse(device='cpu'):
    nbnet = NbNet()
    nbnet.load_state_dict(torch.load("./utils/param/local_inverse.pt", map_location = device))
    nbnet = nbnet.eval().to(device)
    return nbnet    


def get_vit():
    ViT = ViTs_face(
                loss_type='CosFace',
                GPU_ID='0',
                num_class=93431,
                image_size=112,
                patch_size=8,
                ac_patch_size=12,
                pad=4,
                dim=512,
                depth=20,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
    return ViT


def target_FRS(target_FRS='t1', device='cpu'):
    if target_FRS=='t1':
        model = iresnet100(pretrained=False, progress=True)
        model.load_state_dict(torch.load('./utils/param/t1_cosface.pth', map_location = device))
    elif target_FRS=='t2':
        model = get_vit()
        model.load_state_dict(torch.load('./utils/param/t2_vit.pth', map_location = device))        
    elif target_FRS=='t3':
        onnx_model_path = './utils/param/t3_arcface.onnx'
        model = convert(onnx_model_path).eval().to(device)
    elif target_FRS=='t4':
        params = torch.load("./utils/param/t4_sphereface.pth")
        params = dict([(n[7:], p) for n, p in params.items()])
        model = sfnet20_deprecated()
        model.load_state_dict(params, strict=False)
    elif target_FRS=='t5':
        model = InceptionResnetV1(pretrained='casia-webface')
    else:
        ValueError('...invalid input...')
    model = model.eval().to(device)
    return model
    

def blackbox(bb_FRS, bb_th, bb_imgsize, transfer_imgsize, targetimg, query, device):
    decisions = torch.zeros(query.size(0))
    if bb_imgsize!=transfer_imgsize:
        target_f = bb_FRS(F.interpolate(targetimg, transfer_imgsize, mode='nearest').to(device))
        query_f = bb_FRS(F.interpolate(query, transfer_imgsize, mode='nearest').to(device))        
    else:
        target_f = bb_FRS(targetimg.to(device))
        query_f = bb_FRS(query)

    coss = F.cosine_similarity(target_f, query_f)
    angles = torch.acos(coss)*180/torch.pi
    for i in range(query.size(0)):
        if angles[i] <= bb_th:
            decisions[i] = 1
        else:
            decisions[i] = 0

    return decisions, coss