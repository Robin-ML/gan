# Auxiliary Classifier GAN
 These networks help predict the class labels by generating visual images. Such networks are also called visual analogs (VAEs) and generative adversarial networks (GANs). The generative part of the networks tries to generate images that look realistic while the discriminative part tries to differentiate those images from real ones. Auxiliary Classifier GANs (AC-GANs) differ from regular GANs because the class labels are auxiliary information to generate realistic images. In such a framework, the input images from the real class and fake class is fed into the network. AC-GANs have been shown to generate high-resolution images in a way that human beings can't tell the difference between the real and fake images.

A GAN implementation based on the paper [*Conditional Image Synthesis With Auxiliary Classifier GANs*](https://arxiv.org/abs/1610.09585)

