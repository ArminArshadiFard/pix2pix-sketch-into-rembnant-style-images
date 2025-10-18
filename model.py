import torch
import torch.nn as nn


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
            ConvBlock(features * 4, features * 8, kernel_size=3, stride=2, padding=1),
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
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.final(x))


class WGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512, 512]):  # in_channels=6 for real/fake image pairs
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, len(features)):
            layers.append(
                nn.Conv2d(features[i - 1], features[i], kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.InstanceNorm2d(features[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1))  # No sigmoid for WGAN
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Gradient penalty for WGAN-GP
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(batch_size, 1, *d_interpolates.shape[2:], device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Example training loop (simplified, assumes paired dataset)
def train_wgan(generator, discriminator, dataloader, device, n_epochs=100, lambda_gp=10):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0, 0.9))

    for epoch in range(n_epochs):
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            input_imgs, target_imgs = input_imgs.to(device), target_imgs.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            fake_imgs = generator(input_imgs)
            real_pair = torch.cat((input_imgs, target_imgs), dim=1)  # Concatenate input and target
            fake_pair = torch.cat((input_imgs, fake_imgs), dim=1)

            real_validity = discriminator(real_pair)
            fake_validity = discriminator(fake_pair)
            gradient_penalty = compute_gradient_penalty(discriminator, real_pair, fake_pair, device)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train Generator (every 5 discriminator iterations)
            if i % 5 == 0:
                optimizer_G.zero_grad()
                fake_imgs = generator(input_imgs)
                fake_pair = torch.cat((input_imgs, fake_imgs), dim=1)
                g_loss = -torch.mean(discriminator(fake_pair))  # WGAN generator loss
                g_loss.backward()
                optimizer_G.step()

        print(f"Epoch [{epoch + 1}/{n_epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = WGANDiscriminator().to(device)