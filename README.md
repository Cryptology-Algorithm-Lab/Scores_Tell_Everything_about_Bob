# Scores Tell Everything about Bob: Non-adaptive Face Reconstruction on Face Recognition Systems


## Abstract
This code is the implementation for the paper "Scores Tell Everything about Bob: Non-adaptive Face Reconstruction on Face Recognition Systems."

In this paper, we present the first practical score-based face reconstruction and impersonation attack against three commercial face recognition system(FRS) APIs, as well as five commonly used pre-trained open-source FRSs. Our attack is carried out in the black-box FRS model, where the adversary has no knowledge of the FRS (underlying models, parameters, template databases, etc.), except for the ability to make a limited number of similarity score queries. Notably, the attack is straightforward to implement, requires no trial-and-error guessing, and uses a small number of non-adaptive score queries.

In this code, we provide the implementation for our non-adaptive face reconsruction against five pre-trained open-source FRSs. We also provide our pre-generated orthogonal face set(OFS): a precomputed approximate basis set of human-like face images that enables us to get meaningful similarity scores from a small number of non-adaptive queries.


## Quick Start
We provide a guied for this code in the `Scores Tell Everything about Bob.ipynb`. 
For a quick start, we recommend that you follows the step in this file.

- The pre-trained local and target FRSs are in the `/utils/param` including the inverse model of the local FRS.
- We provide our pre-generated OFS in the `sample_OFS`. Instead of the pre-generated OFS, you can generate your own OFS following `GenOFS.py` if you want. You should prepare a pre-trained local FRS, and a dataset for extracting face tempaltes. 

```
sample_LFW                                       # Three target images in LFW dataset, shown in the Table 1 of our paper.
sample_OFS                                       # Pre-generated OFS containing 99 numbers of images.
utils
└── param                                       # Pre-trained FRSs.  
    └── local.pt
    └── local_inverse.pt
    └── t1_cosface.pth
    └── t2_vit.pth
    └── t3_arcface.onnx
    └── tt_sphereface.pth
└── dataset
    └── lfw.bin
Scores Tell Everything about Bob.ipynb           # step-by-step
GenOFS.py                                        # Implementation for GenOFS.
Attack.py                                        # Implementation for Quick Start.
README.md
Reconstruction.py                                # Implementation for our non-adaptive face reconstruction.
```

- Pre-trained model : https://drive.google.com/file/d/1HxuQ3X7Mw7hu2cq-pWCdrRO5AMYGvQ8w/view?usp=drive_link