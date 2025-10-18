# Derm2Real: Clinical Skin Lesion Synthesis from Dermoscopic Images

![Example](assets/example.png)

This project implements a **custom Pix2Pix-style conditional GAN** to translate dermoscopic skin lesion images into realistic clinical photographs — enabling better tele-dermatology and patient communication.

## Why It Matters
- Dermoscopic images are **hard for patients to interpret**
- Clinical-style photos improve **doctor-patient communication**
- Built entirely from scratch — **no copied code**

## Dataset
- **HAM10000** (1,200+ paired images)
- Paired by lesion ID using metadata

## Architecture
- **Generator**: U-Net + Residual blocks
- **Discriminator**: PatchGAN
- **Loss**: L1 + GAN loss

## Results
| Input (Dermoscopic) | Output (Synthetic Clinical) |
|---------------------|-----------------------------|
| ![Input](assets/input.jpg) | ![Output](assets/output.jpg) |

## Run It
```bash
python data/prepare_ham10000.py
python src/train.py
python src/demo.py