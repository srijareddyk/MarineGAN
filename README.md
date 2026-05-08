# MarineGAN: Deep Convolutional GAN for Marine Animal Image Generation


## Project Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch using TensorFlow/Keras to generate realistic 128×128 RGB images of marine animals, specifically focusing on fish species. The model was trained on a subset of the [Kaggle Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data), which contains approximately 13,000 images across 23 marine species classes.

The project explores various training strategies, implements systematic hyperparameter optimization, and includes advanced features such as latent space interpolation and experiment tracking with Weights & Biases.

## Extra Criteria Implemented

This project fulfills:

### 1. **Hyperparameter Optimization**
- Implemented systematic grid search over 9 hyperparameter combinations
- Searched over discriminator learning rate (0.0001, 0.0002, 0.0003) and label noise (0.0, 0.05, 0.1)
- Used 100-image mini-dataset for rapid evaluation (15 epochs per configuration)
- Tracked and compared all experiments to identify optimal settings
- **Best configuration:** lr_d=0.0002, noise_param=0.1

### 2. **Experiment Tracking**
- Integrated Weights & Biases (W&B) for comprehensive experiment tracking
- Logged training metrics (discriminator loss, generator loss, learning rates) in real-time
- Tracked generated image samples at each epoch

### 3. **Latent Space Exploration**
- Implemented latent vector interpolation between random points
- Generated interpolation grids showing smooth transitions between generated fish
- Created animated GIF visualizations of latent space walks

---


## Dataset Preparation

### Source Data
- **Dataset:** [Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data)
- **Original size:** ~13,000 images across 23 marine species classes
- **Resolution:** ~300×300 pixels (varies)

### Preprocessing Steps (`preprocessing.ipynb`)

1. **Class Selection:** Selected fish-related classes from the full dataset:
   - Fish, Sharks, Stingrays, Eels, and related species
   - Excluded non-fish marine animals for visual consistency

2. **Image Validation:**
   - Validated all images using PIL to remove corrupted files
   - Removed unreadable or invalid image files

3. **Dataset Organization:**
   - Copied selected images to `fish-dataset/fish/` directory
   - Prefixed filenames with original class names for traceability
   - Final dataset: **~3,983 valid fish images**

4. **Runtime Preprocessing:**
   - Center-crop and resize to 128×128 RGB
   - Normalize pixel values to [-1, 1] range for GAN training
   - Batch size: 64 images per batch


## Model Architecture

### Generator Network

The generator transforms a 100-dimensional random latent vector into a 128×128×3 RGB image through progressive upsampling:

```
Input: Random vector z ∈ ℝ¹⁰⁰

1. Reshape → (1, 1, 100)
2. ConvTranspose2D (512 filters, 4×4 kernel) → (4, 4, 512)
   - BatchNorm + LeakyReLU(0.2)
3. ConvTranspose2D (256 filters, 4×4 kernel, stride 2) → (8, 8, 256)
   - BatchNorm + LeakyReLU(0.2)
4. ConvTranspose2D (128 filters, 4×4 kernel, stride 2) → (16, 16, 128)
   - BatchNorm + LeakyReLU(0.2)
5. ConvTranspose2D (64 filters, 4×4 kernel, stride 2) → (32, 32, 64)
   - BatchNorm + LeakyReLU(0.2)
6. ConvTranspose2D (32 filters, 4×4 kernel, stride 2) → (64, 64, 32)
   - BatchNorm + LeakyReLU(0.2)
7. ConvTranspose2D (3 filters, 4×4 kernel, stride 2) → (128, 128, 3)
   - Tanh activation (output range [-1, 1])

Total Parameters: ~3.5M
```

### Discriminator Network

The discriminator classifies images as real or fake through progressive downsampling:

```
Input: Image (128, 128, 3)

1. Conv2D (64 filters, 4×4 kernel, stride 2) → (64, 64, 64)
   - LeakyReLU(0.2)
2. Conv2D (128 filters, 4×4 kernel, stride 2) → (32, 32, 128)
   - BatchNorm + LeakyReLU(0.2)
3. Conv2D (256 filters, 4×4 kernel, stride 2) → (16, 16, 256)
   - BatchNorm + LeakyReLU(0.2)
4. Conv2D (512 filters, 4×4 kernel, stride 2) → (8, 8, 512)
   - BatchNorm + LeakyReLU(0.2)
5. Conv2D (512 filters, 4×4 kernel, stride 2) → (4, 4, 512)
   - BatchNorm + LeakyReLU(0.2)
6. Conv2D (1 filter, 4×4 kernel) → (1, 1, 1)
   - Flatten → single logit score

Total Parameters: ~2.8M
```

### Training Objectives

- **Discriminator Loss:** Binary cross-entropy with label smoothing/noise
  - Real images: target = 1.0 ± noise
  - Fake images: target = 0.0 ± noise

- **Generator Loss:** Binary cross-entropy
  - Fake images: target = 1.0 (fool the discriminator)

---

## Training Journey: Experiments and Learnings

### Experiment 1: Initial 5-Epoch Baseline
**Objective:** Quick feasibility test

**Configuration:**
- Epochs: 5
- Learning rates: lr_g = lr_d = 0.0002
- No special techniques

**Results:**
- Poor image quality: Generated images were mostly noise
- Generator loss did not converge meaningfully
- Discriminator dominated too quickly

What I understood from this experiment is that GANs need substantially more training time than 5 epochs to produce coherent outputs.



### Experiment 2: Grid Search for Optimal Hyperparameters `gridsearch.ipynb`
**Objective:** Find best learning rate and label noise settings

**Configuration:**
- Mini-dataset: 100 images (for speed)
- Epochs per trial: 15
- Parameter grid:
  - Discriminator LR: [0.0001, 0.0002, 0.0003]
  - Label noise: [0.0, 0.05, 0.1]
  - Generator LR: Fixed at 0.0002
- Total combinations: 9

**Results:**
| lr_d   | noise | best_g_loss | train_time |
|--------|-------|-------------|------------|
| 0.0002 | 0.1   | **0.7767**  | 38.7s      |
| 0.0002 | 0.05  | 0.8234      | 37.2s      |
| 0.0001 | 0.1   | 0.9156      | 39.1s      |

**Best Configuration:**
- Generator LR: 0.0002
- Discriminator LR: 0.0002
- Label Noise: 0.1
---

### Experiment 3: 80-Epoch Training with Best Hyperparameters
**Objective:** Train full model with optimal settings from grid search

**Configuration:**
- Full dataset: ~3,983 images
- Epochs: 80
- lr_g = lr_d = 0.0002
- Label noise: 0.05
- Standard DCGAN training setup 

**Results:**
- Generated recognizable fish-like/ocean-like images
- Images showed coherent shapes and textures
- Stable training throughout 80 epochs
- Produced the best overall visual quality across all experiments
- image: `results_80epochs/output.png`

What I understood from this experiment is that training longer with the optimal hyperparameters from grid search was enough to produce decent outputs. 

---

### Experiment 4: Early Stopping Attempt
**Objective:** Prevent overfitting using validation-based early stopping

**Configuration:**
- Added discriminator learning rate decay
- Added validation-based early stopping
- Patience: 10 epochs
- Min delta: 0.01
- Monitoring: Generator loss

**Results:**
- This experiment failed, the training stopped at epoch 23
- Generated images were poor quality (`results_earlystopping/epoch 23.png`)
- Model had not converged yet

I understood GANS do not need traditional early stopping
This experiment introduced additional training complexity (learning rate decay + early stopping), but neither improved performance. Early stopping was especially problematic because GAN losses naturally oscillate during adversarial training.

Therefore, I removed this technique entirely.

---

### Experiment 5: Extended 200-Epoch Training with LR Decay
**Objective:** Test whether longer training and discriminator LR decay could further improve results

**Configuration:**
- Epochs: 200
- Same base architecture as the successful 80-epoch run
- Added discriminator learning rate decay

**Configuration:**
- Epochs: 200
- All other settings same as 80-epoch run
- Hypothesis: More training gives better images

**Results:**
- I was actually surprised by the results. Image quality decreased after ~100 epochs
- Generated images became more blurry and abstract
- Loss curves showed continued oscillation but quality declined (`results_200epochs/loss_curves.png`)

After looking at the the loss curves and generated samples, I assumed the following reasons:

1. **Mode Collapse Onset:**
   The generator may have started converging to a limited set of similar outputs. Reduced diversity in generated samples after epoch 100. The discriminator might have adapted to these patterns, creating adversarial pressure. 

2. **Discriminator Over-Regularization:**
   By epoch 200, discriminator LR had decayed to minimum (0.00005) but the generator LR remained at 0.0002 which is an imbalance in learning rate. Generator may have "drifted" without strong discriminator feedback

3. **Lack of Data Augmentation:**
    I used a fixed dataset of ~4,000 images. The generator could have memorized common patterns.What I understod is that more training does not necessarily lead to a better model. There exists an optimal training duration (around 80 epochs for this dataset), beyond which quality can degrade. 
---

### Final Training Recommendation

Use the 80-epoch model in `training.ipynb`:
- Best image quality across all experiments
- Stable training without additional complexity
- No learning rate scheduling required
- Produced the most recognizable and diverse fish images

The later experiments in `improved_training.ipynb` explored LR decay and early stopping, but neither improved final image quality.

---

## How to Run

### Prerequisites

```bash
# Create a virtual environment (recommended)
python -m venv gan_env
source gan_env/bin/activate  

# Install dependencies
pip install tensorflow numpy matplotlib pillow wandb
```

### Step 1: Download and Prepare Dataset

1. Download the [Sea Animals Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data) from Kaggle
2. Unzip so you have an `archive/` folder with class subdirectories
3. Run preprocessing:

```bash
jupyter notebook preprocessing.ipynb
```

This creates the `fish-dataset/` directory with ~3,983 fish images.

### Step 2: (Optional) Run Hyperparameter Grid Search

```bash
jupyter notebook gridsearch.ipynb
```

This takes ~20-30 minutes and tests 9 hyperparameter combinations on 100 images. Results are logged to Weights & Biases.

### Step 3: Train the Full Model

**Option A: Train the Recommended 80-Epoch Model**

1. Open `improved_training.ipynb`
2. Set `EPOCHS_FULL_TRAIN = 80` (line 33)
3. Run all cells


**Option B: Reproduce the 200-Epoch Experiment**

1. Open `improved_training.ipynb`
2. Set `EPOCHS_FULL_TRAIN = 200` (line 33)
3. Run all cells

**Note:** The 200-epoch run demonstrates the degradation phenomenon but does NOT produce the best results.

### Step 4: Generate Images and Explore Latent Space

After training completes, the notebook includes cells to:
- Generate random samples
- Create latent space interpolation grids
- Generate interpolation GIFs
- Plot training loss curves

---

## Weights & Biases Integration

To enable experiment tracking:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Login via terminal:
   ```bash
   wandb login
   ```
3. Set `USE_WANDB = True` in the notebooks (default)
4. View real-time training metrics, generated samples, and hyperparameter comparisons in your W&B dashboard

To disable W&B:
```python
USE_WANDB = False
```

---

### Latent Space Interpolation 

The latent space exploration demonstrates that the generator learned a smooth, continuous manifold. (`interpolation_grid.png` and `interpolation-gif.gif`)

### Training Dynamics

**80-Epoch Training:**
- Generator loss: Converged to ~2.5-3.0 range
- Discriminator loss: Stable around 0.25-0.35
- Discriminator LR: Decayed from 0.0002 → 0.00005

**200-Epoch Training:**
- Generator loss: Continued oscillation around 3.0-4.0
- Discriminator loss: Stable but image quality degraded
- Evidence of mode collapse after epoch 100

---

## Challenges

### Challenge 1: Poor Initial Image Quality

In my first 5-epoch experiment, the generator produced mostly noisy and incoherent outputs. I increased the training duration to 80 epochs and performed a systematic hyperparameter search to identify better training settings.

### Challenge 2: Discriminator Dominance

I observed that the discriminator was learning much faster than the generator, which caused generator training to stall. I introduced label noise (0.05–0.1) to prevent the discriminator from becoming overly confident. I also experimented with discriminator learning rate decay in later runs to reduce its dominance during training.

### Challenge 3: Misjudged Early Stopping

I initially applied early stopping, but training terminated at epoch 23 before the model had learned meaningful image representations. I realized that GANs behave very differently from supervised learning models. Generator and discriminator losses naturally oscillate during adversarial training. Because of this, I removed early stopping entirely and instead relied on visual inspection and W&B monitoring to evaluate progress

### Challenge 4: Unexpected Degradation at 200 Epochs

When I extended training from 80 epochs to 200 epochs, the generated image quality unexpectedly became worse. Training for longer does not always improve results. I included the 200-epoch degradation as an important learning outcome rather than treating it as a failed experiment. 