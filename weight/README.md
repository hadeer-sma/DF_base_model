---
license: cc-by-nc-4.0
language:
- en
---
Pretrained model for <strong>Deepfake Video Detection Using Generative Convolutional Vision Transformer (GenConViT)</strong> paper.

<strong>GenConViT Model Architecture</strong>

The GenConViT model consists of two independent networks and incorporates the following modules:

    Autoencoder (AE),
    Variational Autoencoder (VAE), and
    ConvNeXt-Swin Hybrid layer

GenConViT is trained using Adam optimizer with a learning rate of 0.0001 and weight decay of 0.0001.

GenConViT is trained on the DFDC, FF++, and TM datasets.

GenConViT model has an average accuracy of 95.8% and an AUC value of 99.3% across the tested datasets (DFDC, FF++, and DeepfakeTIMT, Celeb-DF (v2)).

code link: https://github.com/erprogs/GenConViT
