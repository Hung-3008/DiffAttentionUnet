
# Diff-UNet

Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation. Submitted to [MICCAI2023](https://conferences.miccai.org/2023/en/).

[https://arxiv.org/pdf/2303.10326.pdf](https://arxiv.org/pdf/2303.10326.pdf)

We design the Diff-UNet applying diffusion model to solve the 3D medical image segmentation problem.

Diff-UNet achieves more accuracy in multiple segmentation tasks compared with other 3D segmentation methods.

![](/imgs/framework.png)

## dataset 
We release the codes which support the training and testing process of two datasets, BraTS2020 and BTCV.

BraTS2020(4 modalities and 3 segmentation targets): https://www.med.upenn.edu/cbica/brats2020/data.html

BTCV(1 modalities and 13 segmentation targets): https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

Once the data is downloaded, you can begin the training process. Please see the dir of BraTS2020 and BTCV.