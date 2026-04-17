## Title: Marine Animal Image Generation using DCGANs

**Image Source:** This project uses a publicly available Kaggle Marine Animal dataset containing approximately 13,000 images across 23 classes, including fish, jellyfish, octopus, whales, crabs, and other marine species.The images are generally centered and have an original resolution of around 300×300 pixels. For training efficiency, all images will be center-cropped and downsampled to 128×128 RGB. To ensure visual consistency and stable training, the initial model will focus primarily on fish species, with the possibility of extending to additional species if time permits. The goal of this project is to generate realistic images of marine animals that capture variations in shape, texture, colour and anatomical structure.

**Model Architecture:** This project will implement a Deep Convolutional Generative Adversarial Network (DCGAN) trained from scratch. The generator will take a random latent vector sampled from a standard normal distribution and transform it through a series of transposed convolutional layers to produce a 128×128 RGB image. The discriminator will take an image as input and classify it as either real or generated using a series of convolutional layers.

**Extra Criteria:** This project includes several enhancements beyond the baseline GAN implementation. First, latent space exploration will be performed by interpolating between latent vectors to visualize smooth transitions between generated marine animals.

Second, an interactive web-based interface will be developed. This interface will allow users to generate random marine animals and control outputs using a seed value.

Third, systematic hyperparameter tuning will be conducted by experimenting with different learning rates, latent dimensions, and batch sizes(if time permits). These experiments will be tracked and compared to analyze their impact on image quality and diversity.

Finally, experiment tracking and reproducibility will be ensured using tools such as Weights & Biases to log training progress, generated samples, and configurations.

 

This project can be used for data augmentation in marine classification tasks, simulation of biodiversity patterns and procedural content generation for applications such as games and educational tools.

link to dataset: https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data 
