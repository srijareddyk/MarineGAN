# MarineGAN
Marine Animal Image Generation using DCGANs

Marine animal image generation using a **Deep Convolutional GAN (DCGAN)** trained from scratch in **TensorFlow / Keras**. The pipeline follows the course proposal: **128×128 RGB**, **fish-focused** subset from the Kaggle Sea Animals dataset, with optional extras such as **latent interpolation**, **seeded sampling**, **hyperparameter experiments**, and **Weights & Biases** logging.

## Dataset

- **Source:** [Sea Animals Image Dataset (Kaggle)](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data)
- Download and unzip so you have an **`archive/`** folder whose **immediate subfolders are class names** (e.g. `Fish`, `Sharks`, …).

### Fish subset used for training

**`preprocessing.ipynb`** builds **`fish-dataset/`**:

- Copies selected fish-related classes into one folder with prefixed filenames.
- Drops unreadable images (validated with PIL).
- Rearranges files into **`fish-dataset/fish/`** so image loaders see a standard **class folder** layout (`fish-dataset/<class_name>/*.jpg`).

Training expects **`DATA_ROOT = "fish-dataset"`** (**`training.ipynb`**).


