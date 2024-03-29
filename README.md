# Scores Tell Everything about Bob: Non-adaptive Face Reconstruction on Face Recognition Systems

- Sunpill Kim, Yong Kiam Tan, Bora Jeong, Soumik Mondal, Khin Mi Mi Aung, and Jae Hong Seo
- Accepted at IEEE S&P 2024

## Abstract
This code is the implementation for the paper "Scores Tell Everything about Bob: Non-adaptive Face Reconstruction on Face Recognition Systems."

In the paper, we present the first practical score-based face reconstruction and impersonation attack against three commercial face recognition system(FRS) APIs, as well as five commonly used pre-trained open-source FRSs. Our attack is carried out in the black-box FRS model, where the adversary has no knowledge of the FRS (underlying models, parameters, template databases, etc.), except for the ability to make a limited number of similarity score queries. Notably, the attack is straightforward to implement, requires no trial-and-error guessing, and uses a small number of non-adaptive score queries.

In this code, we provide an implementation of our non-adaptive face reconsruction attack. We also provide a pre-generated orthogonal face set (OFS): a precomputed approximate basis set of human-like face images that enables us to get meaningful similarity scores from a small number of non-adaptive queries.

## Quick Start
We provide a starter guide for this code in `Scores Tell Everything about Bob.ipynb`.

The required packages can be installed using `pip`:

```
pip install -r requirements.txt
```

However, the T2 model script has to be installed manually:

```
https://github.com/zhongyy/Face-Transformer (follow instructions for ViTs_face)
```

The LFW dataset and model parameters (local, inverse, T1-T4) can be found at:

```
https://drive.google.com/file/d/1HxuQ3X7Mw7hu2cq-pWCdrRO5AMYGvQ8w/view?usp=drive_link
```

The above compilation is proivded for your convenience. The original links are here: [T1](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215582&cid=4A83B6B633B029CC) on [Insightface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), [T2](https://drive.google.com/drive/folders/1tKjPdDz9WiD-dCjHnkdnyLSs9HS9XUGW) on [Face-Transformer](https://github.com/zhongyy/Face-Transformer?tab=readme-ov-file), [T3](https://drive.google.com/file/d/1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg/view) on [Insightface](https://github.com/deepinsight/insightface/tree/master/model_zoo), [T4](https://drive.google.com/file/d/1ZO3clpW_NHTybOgXIhrA7Kid4OIpQnrG/view) on [opensphere](https://github.com/ydwen/opensphere?tab=readme-ov-file), and [LFW dataset](https://vis-www.cs.umass.edu/lfw/).

After downloading, copy the `param` folder to `utils/param` and the `dataset` folder to `utils/dataset`.

We provide our pre-generated OFS in the `sample_OFS` folder. Instead of the pre-generated OFS, you can generate your own OFS following `GenOFS.py` if you wish. You should prepare a pre-trained local FRS, and a dataset for extracting face tempaltes. 


To facilitate reproducibility, we also provide all 83,000 images reconstructed from our attack against various targets. All reconsturcted images ('recon' folder in below) can be found at:
```
https://zenodo.org/records/10663045
```

The final directory structure should look like this:

```
sample_LFW                                      # Three target images in LFW dataset, shown in the Table 1 of our paper.
sample_OFS                                      # Pre-generated OFS containing 99 images.
recon                                           # All images reconstructed from our attack against various targets.
└── LFW                                            
└── AGE                                         
└── CFP                                         
└── commercial_raw                              # All scores from commercial FRSs.  
utils
└── param                                       # Pre-trained FRSs.  
    └── local.pt
    └── local_inverse.pt
    └── t1_cosface.pth
    └── t2_vit.pth
    └── t3_arcface.onnx
    └── t4_sphereface.pth
└── dataset
    └── lfw.bin
Scores Tell Everything about Bob.ipynb          # Step-by-step quickstart
GenOFS.py                                       # Implementation for GenOFS.
Attack.py                                       # Implementation for Quick Start.
Reconstruction.py                               # Implementation for our non-adaptive face reconstruction.
Example.ipynb                                   # Example
Check Reconstructed Image.ipynb                 # Reproduce the attack success rate table using the reconstructed images.
README.md
```

## Citation
```
@inproceedings{kim2024scores,
  title={Scores Tell Everything about Bob: Non-adaptive Face Reconstruction on Face Recognition Systems},
  author={Kim, Sunpill and Tan, Yong Kiam and Jeong, Bora and Mondal, Soumik and Khin, Mi Mi Aung and Seo, Jae Hong},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={161--161},
  year={2024},
  organization={IEEE Computer Society}
}
```
