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
