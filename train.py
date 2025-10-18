import torch
import torch.nn as nn
from sympy import ground_roots, discrete_log
from torch import device
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import yaml
import os
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


class Paired_Img_Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.a_dir = self.root_dir / "A"
        self.b_dir = self.root_dir / "B"

        if not self.a_dir.exists() or not self.b_dir.exists():
            raise FileNotFoundError(f"Directory missing: {self.a_dir} or {self.b_dir}")

        self.a_files = sorted(list(self.a_dir.glob("*.jpg")))
        self.b_files = sorted(list(self.b_dir.glob("*.jpg")))

        if not self.a_files or not self.b_files:
            raise ValueError(f"No .jpg files found in {self.a_dir} or {self.b_dir}")

        # Validate paired images
        if len(self.a_files) != len(self.b_files):
            raise ValueError(f"Mismatched number of images: {len(self.a_files)} in A, {len(self.b_files)} in B")

        print(f"Loaded {len(self.a_files)} image pairs from {self.root_dir}")

    def __len__(self):
        return len(self.a_files)

    def __getitem__(self, idx):
        sketch_img = Image.open(self.a_files[idx]).convert("RGB")
        real_img = Image.open(self.b_files[idx]).convert("RGB")
        if self.transform:
            sketch_img = self.transform(sketch_img)
            real_img = self.transform(real_img)
        return sketch_img, real_img


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(features, features * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
        ])
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList([
            ConvBlock(features * 4, features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(features * 2, features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.final = nn.Conv2d(features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        assert x.shape[-1] == 256 and x.shape[-2] == 256, f"Input must be 256x256, got {x.shape[-2:]}!"
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, len(features)):
            layers.append(
                nn.Conv2d(features[i - 1], features[i], kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(features[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # ← Standard GAN


def train(config_path="../config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimized transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = Paired_Img_Dataset(config["data"]["root_dir"], split="train", transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    gen = Generator(in_channels=3, features=64).to(device)
    disc = Discriminator(in_channels=6).to(device)

    gen_optim = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_optim = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_generator_loss = float('inf')
    history = {'d_loss': [], 'g_loss': []}

    print(f"training: {len(dataset)} images, batch_size={config['data']['batch_size']}")

    for epoch in range(config["training"]["epochs"]):
        epoch_d_loss = 0
        epoch_g_loss = 0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
            leave=True,
            colour = 'green'
        )

        for idx, (sketch, ground_truth) in progress_bar:
            sketch = sketch.to(device, non_blocking=True)  # ← Faster
            ground_truth = ground_truth.to(device, non_blocking=True)

            # Train Discriminator
            generated_img = gen(sketch)
            disc_real = disc(torch.cat((sketch, ground_truth), 1))
            disc_fake = disc(torch.cat((sketch, generated_img.detach()), 1))

            how_fake_isit = bce(disc_real, torch.ones_like(disc_real))
            how_real_isit = bce(disc_fake, torch.zeros_like(disc_fake))

            disc_loss = (how_fake_isit + how_real_isit) / 2

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # Train Generator
            disc_fake = disc(torch.cat((sketch, generated_img), 1))
            gen_fake_loss = bce(disc_fake, torch.ones_like(disc_fake))
            l1_loss = l1(generated_img, ground_truth) * config["training"]["lambda_l1"]

            g_loss = gen_fake_loss + l1_loss

            gen_optim.zero_grad()
            g_loss.backward()
            gen_optim.step()

            epoch_d_loss += disc_loss.item()
            epoch_g_loss += g_loss.item()
            progress_bar.set_postfix({
                'D_Loss': f'{disc_loss.item():.4f}',
                'G_Loss': f'{g_loss.item():.4f}'
            })

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)

        # Save the best model so far
        if avg_g_loss < best_generator_loss:
            best_generator_loss = avg_g_loss
            torch.save(gen.state_dict(), "checkpoints/best_gen.pth")
            torch.save(disc.state_dict(), "checkpoints/best_disc.pth")
            print(f"Model improved (G Loss: {avg_g_loss:.4f}),(D Loss: {avg_d_loss:.4f})")

        # Save images every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                test_sketch = sketch[:4]
                generated_image = gen(test_sketch)
                test_ground_truth = ground_truth[:4]

                comparison = torch.cat([test_sketch, generated_image, test_ground_truth], dim=3)

                utils.save_image(
                    comparison * 0.5 + 0.5,
                    f"{config['training']['results_dir']}/epoch_{epoch + 1}.jpg",
                    nrow=1
                )

    # generating the plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], 'r-', label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['g_loss'], 'b-', label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n Training complete!")

if __name__ == "__main__":
    train()