"""
model.py
Autoencoder model definition for ultrasound image denoising.
"""

import torch
import torch.nn as nn

class UltrasoundAutoencoder(nn.Module):
    def __init__(self):
        super(UltrasoundAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNNAutoencoderLarge(nn.Module):
    def __init__(self):
        super(CNNAutoencoderLarge, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B, 32, 580, 420)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (B, 32, 290, 210)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (B, 32, 290, 210)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (B, 32, 145, 105)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (B, 32, 145, 105)
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B, 32, 290, 210)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (B, 32, 290, 210)
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B, 32, 580, 420)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),   # (B,  1, 580, 420)
            # Linear activation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === GAN Models (imported from Gannote.py) ===
import torch
import torch.nn as nn

class Generator(nn.Module):
    """U-Net Generator for WGAN-GP"""
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator, self).__init__()
        self.enc1 = self._conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self._conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        up4 = self.upconv4(bottleneck)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(merge4)
        up3 = self.upconv3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        output = self.final_conv(dec1)
        output = self.sigmoid(output)
        return output

class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, input_channels=2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.leaky_relu4(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return torch.mean(x)

class WGAN_GP(nn.Module):
    """Complete WGAN-GP model"""
    def __init__(self):
        super(WGAN_GP, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lambda_gp = 10.0
        self.n_critic = 5
    def gradient_penalty(self, real_samples, fake_samples, device):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        alpha = alpha.expand_as(real_samples)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated = interpolated.requires_grad_(True)
        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    def forward(self, x):
        return self.generator(x)

class DenoisingGenerator(nn.Module):
    def __init__(self):
        super(DenoisingGenerator, self).__init__()
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            self._residual_block(64) for _ in range(6)
        ])
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        residual = x
        x = self.relu(self.initial_conv(x))
        for res_block in self.res_blocks:
            identity = x
            x = res_block(x)
            x = self.relu(x + identity)
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.final_conv(x)
        output = self.sigmoid(x + residual)
        return output

class DenoisingDiscriminator(nn.Module):
    def __init__(self):
        super(DenoisingDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv_layers(x)

class DenoisingGAN(nn.Module):
    def __init__(self):
        super(DenoisingGAN, self).__init__()
        self.generator = DenoisingGenerator()
        self.discriminator = DenoisingDiscriminator()
    def forward(self, x):
        return self.generator(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
