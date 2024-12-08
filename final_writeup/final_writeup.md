### Training MAE and DAE MNIST

Our group decided to start our project by training a DAE and MAE on MNIST data since it was small and easy to prototype with. 

#### DAE Structure:
We found an existing DAE from [Udacity's Deep learning course](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Exercise.ipynb). It has a structure as follows:

The encoder has three convolutional layers with ReLU activations, each followed by max-pooling to reduce the spatial dimensions and extract features.
The channels evolve as follows:
Input image (1 channel) → 32 channels → 16 channels → 8 channels.

Each convolution uses a kernel size of 3×3, and padding ensures the dimensions remain consistent before pooling.

The noise is applied with torch.randn with a noise factor of 0.5.

#### MAE Structure
For the MAE, we utilized the existing code written by the original researchers at Facebook Research Group, [published on Github](https://github.com/facebookresearch/mae). 

We made adjustments that included bugfixes and changing some modules so that they would return the attention maps and latent spaces of the model. 

##### Encoder Structure

The encoder processes only the visible (unmasked) patches.
It uses a Vision Transformer (ViT), which applies several layers of transformer blocks to extract high-level features.
A special "classification token" (cls_token) is added to summarize the global information across all patches.

##### Decoder
The decoder receives the latent representation from the encoder and reconstructs the image. It consists of masked tokens, positional embeddings, linear layers, and more transformers.

Masked tokens (representing the missing patches) are added back into the sequence to reconstruct the full set of patches.

Positional embeddings are used to inform the network about the spatial arrangement of patches.

The decoder uses another set of transformer blocks to predict the pixel values of the missing patches.

##### Training Details
The loss function for a MAE compares only the reconstructed (masked) patches against the original patches. Unlike the DAE, it does not compare the entire reconstructed image to the original image, which is why as the original paper notes, the reconstruction of the originally visible patches is noticeably worse as they are not included in the loss function. 

#### Reconstruction images

![dae reconstruction](Reconstructions/dae_reconstruction_original.png)
*Fig x. DAE Reconstruction of noisy input images*

![mae reconstruction](Reconstructions/mae_reconstruction_original.png)
*Fig x. MAE Reconstruction of masked input images*

As we can see with Fig. x and Fig. x, The MAE generally reconstructs much better. The final loss for training the DAE was around 0.532, whereas the final loss for the MAE was only 0.11 for 20 epochs each. 

Train loss: 0.1216
Test loss: 0.1194


### Visualization of the Latent Space

Our hypothesis for why the MAE performs better was that the latent space contained more information that helped distinguish the image. In order to test this theory, we decided to visualize the latent space for both the DAE and the MAE.

We took the latent representations for the DAE and MAE and then visualized them using T-SNE reduction to gain the following graphs:

![dae clustering](Latent_Clustering/dae_latent_clustering.png)
*Fig. X: Visualized the DAE Latent space using T-SNE dimensional reduction*

![mae clustering](Latent_Clustering/mae_latent_clustering.png)
*Fig. X: Visualized the MAE Latent space using T-SNE dimensional reduction*

What's interesting about these images is we can see a cluster effect in the MAE latent space visualization. The latent spaces for images from the same classes are spaced much closer together for the MAE compared to the DAE. While they are not true separated clusters, there is definitely more grouping by class occurring for the MAE compared to the DAE. The radiuses of these latent space clusters were as follows:

| **Class** | **MAE Cluster Radius** | **DAE Cluster Radius** |
|-----------|-------------------------|-------------------------|
| 0         | 4.6971                 | 14.9228                |
| 1         | 3.6383                 | 12.1421                |
| 2         | 11.6399                | 21.0691                |
| 3         | 8.2036                 | 13.7999                |
| 4         | 7.8591                 | 17.9415                |
| 5         | 6.4910                 | 15.8864                |
| 6         | 5.3545                 | 12.4632                |
| 7         | 6.1213                 | 16.9559                |
| 8         | 10.2892                | 19.1732                |
| 9         | 7.7658                 | 16.4148                |

These radii were calculated by first finding the cluster center in the T-SNE reduced latent space. This is simply the average position (mean) of all the latent representations for that class. The radii is then just the average distance of each sample to its cluster center, to give us an idea of how spread out the radii are. We chose to do this for the reduced latent space as opposed to the true latent space because the MAE original latent space had a much higher dimensionality than the DAE latent space. 

These numbers and visualizations show that the MAE number classes in general had a much tighter cluster. These results suggest that the MAE outperforms the DAE in large part due to the latent space information. 

### Visualizing the Attention Maps

![dae heatmaps](Heatmaps/dae_map_4.png)
*Fig. x: DAE Saliency maps*

![mae heatmaps](Heatmaps/mae_map_4.png)
*Fig. x: MAE Attention heatmaps*

The DAE saliency map is computed by measuring how sensitive the latent representation of a denoising autoencoder (DAE) is to changes in each pixel of the input image. Gradients are calculated to find out how much each pixel of the input image contributes to the overall latent representation. This is done by "backtracking" from the latent space through the network to the input image. These gradient magnitudes are normalized to a 0–1 range and displayed as a heatmap, where brighter regions indicate pixels that have a greater impact on the latent space.

The MAE attention map is computed by 
