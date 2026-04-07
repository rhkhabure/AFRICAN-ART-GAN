# African Fabric GAN — Generating Synthetic African Fabric Patterns
- Howard Muchaki 
- Aime Muganga 

## Project Overview

This project trains a Generative Adversarial Network (GAN) to generate synthetic Artistic African fabric images. The model is trained on a dataset of approximately 1,000 African fabric images.

## Why Lightweight GAN?

After experimenting with several GAN architectures including DCGAN, FastGAN, and Progressive GAN, all of which produced blurry, grey, or incoherent images, we switched to **Lightweight GAN** (proposed at ICLR 2021 by Liu et al.).

Lightweight GAN was chosen for three key reasons:

1. **Designed for small datasets** — it includes built-in differentiable augmentation (DiffAugment) that prevents the discriminator from overfitting when training data is limited. With only ~1,000 images, this is critical.
2. **No custom CUDA operations** — unlike StyleGAN2-ADA which requires compiling custom CUDA kernels and caused numerous compatibility errors on Google Colab, Lightweight GAN is pure PyTorch and installs with a single `pip install`.
3. **Fast convergence** — the paper demonstrates convergence in a few hours on a single GPU, even at 1024×1024 resolution with sub-100 images.

## How a GAN Works

A GAN consists of two neural networks competing against each other:

- **Generator (G)** — takes random noise as input and tries to produce images realistic enough to fool the discriminator. It never sees real images directly; it only learns from the discriminator's feedback.
- **Discriminator (D)** — looks at both real images from the dataset and fake images from the generator, and tries to tell them apart. It outputs a score indicating how "real" an image looks.

These two networks are trained simultaneously in a minimax game: the generator tries to minimize the discriminator's ability to detect fakes, while the discriminator tries to maximize it. Over time, the generator learns to produce increasingly realistic images.

In Lightweight GAN specifically, the discriminator also includes a **self-supervised reconstruction branch** — it tries to reconstruct portions of real images, which helps it learn richer features even from small datasets.

## Notebook Structure

1. Load and explore the dataset metadata
2. Download and prepare images
3. Mount Google Drive and set up output folders
4. Train the Lightweight GAN
5. Generate and export results
6. Save the final model

##  Training Pipeline
1. **Data Preprocessing**  
   - Images resized to 256×256  
   - Normalised to [-1, 1]  

2. **Model Architecture**  
   - Generator: Deep convolutional + residual blocks  
   - Discriminator: Spectral normalisation + dropout  

3. **Manual Training Loop**

   We bypassed the `Trainer` class entirely and wrote a manual PyTorch training loop. This approach gives full control over each training step and       avoids all version-related API mismatches. The key components are:

   - **Hinge loss** for both generator and discriminator, which is standard for GANs and more stable than binary cross-entropy
   - **DiffAugment** applied to both real and fake images before passing them to the discriminator, using color, translation and cutout augmentations
   - **Checkpointing every 1,000 steps** to Google Drive so training can resume after a Colab disconnect
   - **Sample images saved every 1,000 steps** so we can visually monitor quality during training 

---

## Training Results — Interpretation

Training ran for **10,000 steps** (~2 hours 38 minutes on a T4 GPU, approximately 1.05 steps/second).

### Loss Values

The two loss values reported at each checkpoint are:

- **G (Generator loss)** — measures how well the generator is fooling the discriminator. Lower is better, but some fluctuation is normal and expected.
- **D (Discriminator loss)** — measures how well the discriminator is distinguishing real from fake. A healthy discriminator loss typically stays in the range of 0.5–2.0.

| Step | G Loss | D Loss | Interpretation |
|------|--------|--------|----------------|
| 1,000 | 0.286 | 1.636 | Early stage — discriminator dominates, generator has not learned much yet |
| 3,000 | 1.483 | 1.125 | Generator improving — losses converging toward each other |
| 5,000 | 1.114 | 0.867 | Training stabilising — both networks competing more evenly |
| 7,000 | 0.443 | 0.714 | Generator gaining ground — producing more convincing images |
| 10,000 | 2.271 | 0.304 | Generator strong — low discriminator loss suggests some mode competition |

### What the Loss Trends Tell Us

The general pattern shows the generator and discriminator reaching a competitive equilibrium around steps 5,000–7,000, which is when the visual quality of generated images was most consistent. The spike in generator loss at step 10,000 with a very low discriminator loss (0.304) indicates the discriminator momentarily gained an edge — this is normal GAN behaviour and does not mean the model has collapsed. The saved samples at steps 3,000, 7,000 and 10,000 were all visually usable.
---

##  Generated Samples
Below are grids of generated outputs at different steps:

### step 001000
<IPython.core.display.Image object><img width="1034" height="1034" alt="image" src="https://github.com/user-attachments/assets/fc8cc66e-33cf-4b60-bc63-c4773659b039" />



### Step 00200
<IPython.core.display.Image object><img width="1034" height="1034" alt="image" src="https://github.com/user-attachments/assets/bdedc220-b7b6-47b5-9cbd-85b5bc39f987" />




### step 00500
<IPython.core.display.Image object><img width="1034" height="1034" alt="image" src="https://github.com/user-attachments/assets/ac262616-8f39-4d56-83da-121a9720818e" />




### step 01000
<IPython.core.display.Image object><img width="1034" height="1034" alt="image" src="https://github.com/user-attachments/assets/349cabcf-a660-4fca-8632-e1c9239633d5" />



---

##  Evaluation
- **Metric**: Frechet Inception Distance (FID)  
- **Implementation**: TorchMetrics `FrechetInceptionDistance`  
- **Interpretation**: Lower FID = more realistic outputs. Current scores remain high, indicating room for improvement.

---




## Conclusion

This notebook demonstrates the end-to-end training of a Lightweight GAN on a custom African fabric image dataset. Starting from raw image URLs, we downloaded and preprocessed approximately 1,000 images, trained a GAN for 10,000 steps, and exported both the generated samples and the trained model weights.

### Key Takeaways

- **Architecture choice matters for small datasets.** DCGAN, FastGAN and Progressive GAN all failed to produce coherent results on ~1,000 images. Lightweight GAN's built-in differentiable augmentation was the critical factor that made training viable at this dataset size.
- **API compatibility requires careful debugging.** The high-level `Trainer` API had version mismatches that caused silent failures. Dropping down to a manual training loop resolved all issues and gave better control over the process.
- **GAN training is inherently unstable.** Loss values fluctuate throughout training — this is expected behaviour, not a sign of failure. Visual inspection of generated samples at regular checkpoints is more informative than loss values alone.

### Limitations and Future Work

- With only 1,000 images and 10,000 training steps, the generated images show recognisable fabric-like patterns but lack the fine detail and variety of real African fabrics. Training on a larger dataset (5,000+ images) for more steps would significantly improve quality.
- The model was trained at 256×256 resolution. Higher resolution (512×512) would produce sharper outputs but requires more GPU memory and training time.
- Future work could explore fine-tuning from a pretrained model (such as one trained on a broader textile or art dataset) rather than training from scratch, which would reduce the data requirements considerably.

---

##  Repository Structure
├── data/                # Dataset (African art images)
├── checkpoints/         # Saved model states
├── samples/             # Generated images per epoch
├── models.py            # Generator + Discriminator definitions
├── train.py             # Training loop
├── evaluate_fid.py      # FID computation
└── README.md            # Project documentation
