from .verification import *
import torch.nn.functional as F


def test_inverse(data_set, backbone, th, gen, batch_size, device, nfolds=1):
    cos = (th-2)/-2
    print('\ntesting verification..')
    data1=data_set[0][0]
    data2=data_set[0][1] #flip
    
    issame_list = data_set[1]
    embeddings_list = [
        
        np.zeros((data1.shape[0], 512)), 
        np.zeros((data2.shape[0], 512))
        
    ]
    time_consumed = 0.0    
    
    ba = 0
    while ba < data1.shape[0]:
        bb = min(ba + batch_size, data1.shape[0])
        count = bb - ba
        _data1 = data1[bb - batch_size: bb]
        _data2 = data2[bb - batch_size: bb]
        
        time0 = datetime.datetime.now()
        img1 = (_data1/255. - 0.5) / 0.5
        img2 = (_data2/255. - 0.5) / 0.5
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        net_out1 = backbone(img1) #feature
        net_out2 = backbone(img2) #flip
        
        f1 = net_out1[0::2]
        f2 = net_out1[1::2] #recon
        f3 = net_out2[0::2]
        f4 = net_out2[1::2]
        

        fake_img = gen(F.normalize(f2))
        fake_img = F.interpolate(fake_img, 112, mode='nearest')
        flip = fake_img.flip(3)
        
        net_out1[1::2] = backbone(fake_img)
        net_out2[1::2] = backbone(flip)
        
        _embeddings1 = net_out1.detach().cpu().numpy()
        _embeddings2 = net_out2.detach().cpu().numpy()

    
        time_now = datetime.datetime.now()
        diff = time_now - time0
        time_consumed += diff.total_seconds()

        embeddings_list[0][ba:bb, :] = _embeddings1[(batch_size - count):, :]
        embeddings_list[1][ba:bb, :] = _embeddings2[(batch_size - count):, :]#feature
        
        ba = bb

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    cosval=embeddings[0::2]*embeddings[1::2]
    cosval=cosval.sum(1)
    truecosval=cosval[issame_list]
    print('# Target images:', len(truecosval))
    print('# Inversion success:', (truecosval>=cos).sum())
    print('[Inversion Success Rate]:',(truecosval>=cos).sum()/len(truecosval))
    return