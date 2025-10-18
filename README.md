# Pix2Pix: Turning Sketch into Remnant style images using GAN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

This project is an implementation of **Pix2Pix conditional GAN** to transform some rough lines into meaningful images in Rembrandt Style images. I used a very small dataset of sketch and real pairs, splited them into sketch and ground truth and then again divided into test and train 

## Dataset
- **Rembrandt** (302 paired images)
- Splited into sketch and actuall drawings
- divided into test and train
- trained for 50 epochs

## Architecture
- **Generator**: Encoder–decoder with 9 residual blocks in the bottleneck  
- **Discriminator**: PatchGAN (70×70 receptive field), not a generic cGAN  
- **Loss Function**: Hybrid objective combining **adversarial loss (BCE)** and **L1 reconstruction loss**
  
> 💡 **Note**: the model was still improving and could have generated better images if trained for longer  
> this is the generator and discriminator losses of my model: **Generator Loss: 3.8117**, **Discriminator Loss: 0.6390**  
> Training for longer will surley improve the results even more
> 

## Results
### Training Curves  
<img src="https://github.com/user-attachments/assets/af23aaa1-5385-4c7a-99dc-905b7b511373" alt="Training Curves" width="800">

### Sample Outputs at 20 , 40 and 50 epochs

<div align="center">
  <img src="https://github.com/user-attachments/assets/5810fef3-f74a-47ce-99ba-c35cfa2e2e44" alt="Epoch 20" width="450">
  <img src="https://github.com/user-attachments/assets/d205fb92-a62c-46fb-bf6e-160119ce95ee" alt="Epoch 40" width="450">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/6b8c73e9-1730-4944-93fb-b6c1da772c61" alt="Epoch 50" width="900">
</div>


