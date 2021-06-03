# Learning-a-Few-shot-Embedding-Model-with-Contrastive-Learning
This repo contains codes for Learning a Few-shot Embedding Model with Contrastive Learning(AAAI2021)

## Abstract
Few-shot learning (FSL) aims to recognize target classes by
adapting the prior knowledge learned from source classes.
Such knowledge usually resides in a deep embedding model
for a general matching purpose of the support and query
image pairs. The objective of this paper is to repurpose
the contrastive learning for such matching to learn a fewshot embedding model. We make the following contributions: (i) We investigate the contrastive learning with Noise
Contrastive Estimation (NCE) in a supervised manner for
training a few-shot embedding model; (ii) We propose a
novel contrastive training scheme dubbed infoPatch, exploiting the patch-wise relationship to substantially improve
the popular infoNCE. (iii) We show that the embedding
learned by the proposed infoPatch is more effective. (iv) Our
model is thoroughly evaluated on few-shot recognition task;
and demonstrates state-of-the-art results on miniImageNet
and appealing performance on tieredImageNet, FewshotCIFAR100 (FC-100). 

## Paper 
[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17047/16854)

## Citation
@inproceedings{liu2021learning,
  title={Learning a Few-shot Embedding Model with Contrastive Learning},
  author={Liu, Chen and Fu, Yanwei and Xu, Chengming and Yang, Siqian and Li, Jilin and Wang, Chengjie and Zhang, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={10},
  pages={8635--8643},
  year={2021}
}

## Pretrain Weight
[ResNet-12](https://drive.google.com/drive/folders/1k7bJrBMucPWB3FVeXq0ay1xQOVTJAHFv?usp=sharing)


