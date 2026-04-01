# African Art GAN Project

##  Overview
This project explores the use of Generative Adversarial Networks (GANs) to generate synthetic African art images.  
The goal is to train a GAN on curated datasets and evaluate realism using **Frechet Inception Distance (FID)**.

---

##  Project Setup
- **Frameworks**: PyTorch, Torchvision
- **Models**: Custom Generator (with residual blocks), Discriminator (with spectral normalisation + dropout)
- **Training**: 75 epochs, TTUR learning rates, instance noise regularisation
- **Evaluation**: FID score computed using TorchMetrics

---

##  Training Pipeline
1. **Data Preprocessing**  
   - Images resized to 256×256  
   - Normalised to [-1, 1]  

2. **Model Architecture**  
   - Generator: Deep convolutional + residual blocks  
   - Discriminator: Spectral normalisation + dropout  

3. **Training Loop**  
   - TTUR (different learning rates for G and D)  
   - Instance noise (decayed over epochs)  
   - Checkpoints saved every epoch  

---

##  Results
- **Training runs**: 10, 30, and 45 epochs  
- **FID scores**:  
  - 10 epochs → ~720 
  - 30 epochs → ~297  
  - 45 epochs → ~550

Interpretation: The Generator learns colour and texture distributions, but realism remains limited.  
Future improvements will focus on dataset augmentation, progressive growing, and advanced architectures (StyleGAN, BigGAN).

---

## 🖼️ Generated Samples
Below are grids of generated outputs at different epochs:

### Epoch 5
*(Insert image here)*

### Epoch 10
*(Insert image here)*

### Epoch 15
*(Insert image here)*

### Epoch 20
*(Insert image here)*

### Epoch 25
*(Insert image here)*

### Epoch 45
*(Insert image here)*

---

## 📈 Evaluation
- **Metric**: Frechet Inception Distance (FID)  
- **Implementation**: TorchMetrics `FrechetInceptionDistance`  
- **Interpretation**: Lower FID = more realistic outputs. Current scores remain high, indicating room for improvement.

---

## 🔮 Future Work
- Progressive growing to higher resolutions (512×512)  
- Larger and more diverse African art datasets  
- Advanced GAN architectures (StyleGAN, BigGAN)  
- Improved regularisation (WGAN‑GP, gradient penalty)  
- Data augmentation for robustness  

---

## 📂 Repository Structure
