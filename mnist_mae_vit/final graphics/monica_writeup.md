### Training MNIST

Our group decided to start our project by training a DAE and MAE on MNIST data since it was small and easy to prototype with. 

#### DAE Structure:
We found an existing DAE from [Udacity's Deep learning course](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Exercise.ipynb). It has a structure as follows:

The encoder has three convolutional layers with ReLU activations, each followed by max-pooling to reduce the spatial dimensions and extract features.
The channels evolve as follows:
Input image (1 channel) → 32 channels → 16 channels → 8 channels.

Each convolution uses a kernel size of 3×3, and padding ensures the dimensions remain consistent before pooling.

The noise is applied with torch.randn with a noise factor of 0.5.

#### MAE Structure