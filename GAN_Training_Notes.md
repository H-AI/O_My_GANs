# How to Train a GAN? Tips and tricks to make GANs work
Below are notes learned and collected from my studies on techniques for GANs training. 

## Read Sources
[ganhacks](https://github.com/soumith/ganhacks)

## Summary of Useful Techniques

### 1. Normalize the inputs
normalize the images between -1 and 1
Tanh as the last layer of the generator output

### 2: A modified loss function
In GAN papers, the loss function to optimize G is min (log 1-D), but in practice folks practically use max log D

because the first formulation has vanishing gradients early on
Goodfellow et. al (2014)
In practice, works well:

Flip labels when training generator: real = fake, fake = real

### 4: Use BN and separate real and facke minibatch
Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.

### 5: Avoid Sparse Gradients: ReLU, MaxPool
the stability of the GAN game suffers if you have sparse gradients
LeakyReLU = good (in both G and D)
For Downsampling, use: Average Pooling, Conv2d + stride
For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
PixelShuffle: https://arxiv.org/abs/1609.05158

### 6: Use Soft and Noisy Labels
Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
Salimans et. al. 2016
make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator

### 8: Use stability tricks from RL
Experience Replay
Keep a replay buffer of past generations and occassionally show them
Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations
All stability tricks that work for deep deterministic policy gradients
See Pfau & Vinyals (2016)

### 9: Use the ADAM Optimizer
optim.Adam rules!
See Radford et. al. 2015
Use SGD for discriminator and ADAM for generator

### 12: If you have labels, use them
if you have labels available, training the discriminator to also classify the samples: auxillary GANs

### 17: Use Dropouts in G in both train and test phase
Provide noise in the form of dropout (50%).
Apply on several layers of our generator at both training and test time
https://arxiv.org/pdf/1611.07004v1.pdf
