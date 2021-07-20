![license MIT](https://img.shields.io/badge/licence-MIT-green)

![main](assets/logo.png)
## GAN
 Generative Adversarial Networks refers to models that take a data distribution and an unknown function in the form of a generator and a discriminator as its input. In particular, for a discriminator that is trained to classify data, a neural network with multiple layers is commonly used. 

 The number of layers is greater than the number of elements in a dataset, so the discriminator is a deep neural network. A discriminator with multiple layers can be described as D(x), where x is a data vector. D(x) represents a distribution over all possible data. 
However, in a discriminator with multiple layers, D(x) is still not the same as the true distribution. That is, we may set a threshold t and decide if a data x belongs to a certain category according to D(x). 
When D(x) = 1, we set x as being in a certain category; when D(x) = 0, we set x as being in another category.
For example, the threshold is set at 0.5 for a two-class classification problem, so we decide that the input belongs to one of the two categories according to the output of D(x). When the input is in one category, its probability is p(1). Similarly, when the input is in another category, its probability is p(0).

 In a generative adversarial network (GAN), the goal is to train a generator that generates new data as similar as possible to the real distribution. We usually set the generator to a distribution as defined by a simple and easily understood model. 
GAN is a generative model that can simulate the distribution of data and then generate data to match a given distribution. Given the fact that it is easy to understand the internal process, the generator model is relatively simple. 
 Many of the generator models are simple distributions, including Gaussian, Laplacian, and Poisson distributions. Therefore, the input of the generator is x, which is a scalar or a vector. The output of the generator is y, which is a scalar or a vector. In the output of the generator, the number of positive values is identical to the number of negative values. If the number of negative values is p, and the number of positive values is n, then the number of elements of the vector y is 1.
There are various forms of GAN implementations source code.

Some versions are newer and more polished, and I generally recommend using simpler implementations as a starting point if you are looking to experiment with the techniques, build upon it, or apply it to novel datasets.

Some of GANs include:
## Table of Contents
  * [Installation](#installation)
  * [GAN Variants](variants)
    + [Auxiliary Classifier GAN](variants/acgan)
    + [Adversarial Autoencoder](variants/aae)
    + [BEGAN](variants/began)
    + [BicycleGAN](variants/bicyclegan)
    + [Boundary-Seeking GAN](variants/bgan)
    + [Cluster GAN](variants/cluster-gan)
    + [Conditional GAN](variants/cgan)
    + [Context-Conditional GAN](variants/ccgan)
    + [Context Encoder](variants/context-encoder)
    + [Coupled GAN](variants/coupled-gan)
    + [CycleGAN](variants/cyclegan)
    + [Deep Convolutional GAN](variants/dcgan)
    + [DiscoGAN](variants/discogan)
    + [DRAGAN](variants/dragan)
    + [DualGAN](variants/dualgan)
    + [Energy-Based GAN](variants/energy-based-gan)
    + [Enhanced Super-Resolution GAN](variants/enhanced-super-resolution-gan)
    + [GAN](variants/gan)
    + [InfoGAN](variants/infogan)
    + [Least Squares GAN](variants/least-squares-gan)
    + [MUNIT](variants/munit)
    + [Pix2Pix](variants/pix2pix)
    + [PixelDA](variants/pixelda)
    + [Relativistic GAN](variants/relativistic-gan)
    + [Semi-Supervised GAN](variants/semi-supervised-gan)
    + [Softmax GAN](variants/softmax-gan)
    + [StarGAN](variants/stargan)
    + [Super-Resolution GAN](variants/super-resolution-gan)
    + [UNIT](variants/unit)
    + [Wasserstein GAN](variants/wgan)
    + [Wasserstein GAN GP](variants/wgan-gp)
    + [Wasserstein GAN DIV](variants/wgan-div)

## Installation
    $ git clone https://github.com/Robin-ML/gan
    $ cd gan
    $ sudo pip3 install -r requirements.txt
