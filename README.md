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
- ** Deployment** : https://african-art-gan-t4ngj82dhbba6mlqqjwn6m.streamlit.app/ link to the streamlite app

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

##  Generated Samples
Below are grids of generated outputs at different epochs:

### Epoch 1
<Figure size 800x800 with 1 Axes><img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/29d63df0-a383-443b-8050-9245ee871915" />


### Epoch 5
<img width="530" height="530" alt="epoch_5" src="https://github.com/user-attachments/assets/2ff8f7e2-1bc5-4165-9705-e211781ba5dc" />



### Epoch 10
<Figure size 800x800 with 1 Axes><img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/62a09c10-0368-43c0-846f-33f55ac2ca4a" />



### Epoch 20
<Figure size 800x800 with 1 Axes><img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/90fc6a6f-e264-4906-94f9-8d77be6db787" />


### Epoch 30
<Figure size 800x800 with 1 Axes><img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/831e3b3a-e9c0-482f-94bc-4cad25d44627" />



### Epoch 45
<Figure size 800x800 with 1 Axes><img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/9f398162-f382-4c40-a24e-66c7e1c15465" />


---

##  Evaluation
- **Metric**: Frechet Inception Distance (FID)  
- **Implementation**: TorchMetrics `FrechetInceptionDistance`  
- **Interpretation**: Lower FID = more realistic outputs. Current scores remain high, indicating room for improvement.

---

## Challenges and Weaknesses

### Struggles
Training Instability  
GANs are notoriously unstable. Losses oscillated heavily, and sometimes the Generator lagged behind the Discriminator, producing poor samples.

High FID Scores  
Even after 75 epochs, FID remained high (>300), showing that realism was limited. Regularization tweaks (TTUR, instance noise) sometimes made diversity better but worsened FID.

Mode Collapse Risk  
At certain epochs, the Generator produced repetitive patterns instead of diverse outputs. This is a common DCGAN issue.

Compute Requirements  
Training at 256×256 resolution was already resource‑intensive. Scaling to higher resolutions (512×512) would require much more compute and memory.

Dataset Limitations  
The dataset size and diversity constrained the Generator’s ability to learn distinctive “African art” features. GANs need large, balanced datasets to capture style.

- Training instability with oscillating losses.
- High FID scores despite longer training.
- Mode collapse leading to repetitive outputs.
- High compute requirements for larger resolutions.
- Dataset limitations restricting stylistic learning.

### Weaknesses of DCGAN
Architecture Simplicity  
DCGAN is a baseline model. It lacks advanced mechanisms (style mixing, attention layers, progressive growing) that modern GANs use to capture complex features.

Resolution Ceiling  
DCGAN struggles beyond 256×256. Outputs at higher resolutions often collapse or blur.

Limited Feature Control  
Unlike StyleGAN, DCGAN doesn’t allow fine‑grained control over attributes (e.g., texture, color, style). This makes it harder to steer outputs toward specific artistic qualities.

Slow Convergence  
DCGAN often requires hundreds of epochs to produce realistic samples, and even then, results may plateau.

- Simple architecture compared to modern GANs.
- Struggles beyond 256×256 resolution.
- No fine‑grained control over generated features.
- Slow convergence and limited realism.


##  Future Work
- Progressive growing to higher resolutions (512×512)  
- Larger and more diverse African art datasets  
- Advanced GAN architectures (StyleGAN, BigGAN)  
- Improved regularisation (WGAN‑GP, gradient penalty)  
- Data augmentation for robustness  

---

##  Repository Structure
├── data/                # Dataset (African art images)
├── checkpoints/         # Saved model states
├── samples/             # Generated images per epoch
├── models.py            # Generator + Discriminator definitions
├── train.py             # Training loop
├── evaluate_fid.py      # FID computation
└── README.md            # Project documentation
