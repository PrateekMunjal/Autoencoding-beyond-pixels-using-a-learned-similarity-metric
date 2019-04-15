# VAE/GAN
A Tensorflow implementation of VAE-GAN, following the paper: [VAE/GAN](https://arxiv.org/abs/1512.09300). The encoder and decoder functions are implemented using fully strided convoluttional layers and transposed convolution layers respectively. The discriminator network has the same architecture as that of encoder with an additional last one layer of its output. As suggested by papers I have implemented Gaussian decoders and Gaussian prior.

## Setup
* Python 3.5+
* Tensorflow 1.9

## Relevant Code Files

File config.py contains the hyper-parameters for VAE/GAN reported results.

File vae-gan.py contains the code to train VAE/GAN model.

Similarly, as the name suggests, file vae-gan_inference.py contains the code to test the trained VAE/GAN model.

## Usage
### Training a model

```
python vae-gan.py
```

### Test a trained model 
 
First place the model weights in model_directory (mentioned in vae-gan_inference.py) and then:
```
python vae-gan_inference.py 
```

## Emprical Observations

* I observed that sometimes the presence of KL-divergence term in the loss of encoder network makes the model training cumbersome.
* The only hyper-parameter I tweaked to alleviate the above issue is weight mutiplied to this KL term. Almost always, the KL-term weight equal to 1/batch_size works.
* Another alternate I tried for Kl weight was taking as a function of epoch i.e sigmoid(epoch). 
* Intuitively, the dynamic Kl weight made more sense as with increasing epochs we increased the weight, therefore the model does not pay attention to KL divergence term in initial iterations. However, one should ask ***why do we want the model to not focus in initial iterations?***
* The reason is that we free the latent space variables in initial iterations to make them learn, meaningful representations responsible for reconstructing the input and with increasing epochs we make the latent distribution close to our prior as we increase KL term weight with epochs.

* ***But why did not we used some other function like exp(epochs)?*** -- It is also a monotonic function.
* While increasing the weight of KL term, we should have some limit else the model may completely focus on this term. Therefore we choose a function which has a saturation on large values of input.

## Model weights

The weights for presented results in this repository are mentioned below which essentially are shared on google drive. 

* [MNIST model weights](https://drive.google.com/drive/folders/16d0OcY5ub_ladisKtFfSqBX4TOejv5la?usp=sharing)
* [CelebA model weights](https://drive.google.com/drive/folders/1G7-wBlxp2CFbhNidUNBxRCG6I-uXusg1?usp=sharing)

## Generations

MNIST            |  Celeb-A
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/MNIST/generations.gif)  |  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/Celeb-A/generations.gif)

## Reconstructions
* For MNIST dataset
  * **At epoch: 1**

  MNIST Original            |  MNIST Reconstruction
  :-------------------------:|:-------------------------: 
  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/MNIST/op-real/original_new_vae_0.png)  |  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/MNIST/op-recons/reconstructed_new_vae0.png)
  * **At epoch: 50**

  MNIST Original            |  MNIST Reconstruction
  :-------------------------:|:-------------------------:
  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/MNIST/op-real/original_new_vae_50.png)  |  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/MNIST/op-recons/reconstructed_new_vae50.png)

* For CelebA dataset
  * **At epoch: 1**
  
  Celeb-A Original            |  Celeb-A Reconstruction
  :-------------------------:|:-------------------------: 
  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/Celeb-A/op-real/orig-img-0.png)  |  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/Celeb-A/op-recons/recons-img-0.png)
  
    * **At epoch: 15**
  
  Celeb-A Original            |  Celeb-A Reconstruction
  :-------------------------:|:-------------------------: 
  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/Celeb-A/op-real/orig-img-14.png)  |  ![](https://github.com/PrateekMunjal/VAE_GAN/blob/master/Celeb-A/op-recons/recons-img-14.png)



