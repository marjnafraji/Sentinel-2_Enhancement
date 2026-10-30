#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pan_G: GAN + SSIM loss for panchromatic (PAN) band generation (PyTorch)
=======================================================================

This script trains a conditional GAN that takes a 4-band multispectral (MS)
Landsat image and generates the corresponding single-band panchromatic (PAN)
image. The trained generator can then be applied to large Sentinel-2 GeoTIFFs
to produce a PAN-like band (optionally at a higher spatial resolution).

Pipeline overview
-----------------
1. Read Landsat GeoTIFFs (bands 0-3 = MS, band 4 = PAN) and cut them into
   128x128 patches.
2. Split patches into train / test sets.
3. Train the GAN:
      Generator     : residual CNN (MS 4-band  -> PAN 1-band)
      Discriminator : CNN classifier (real PAN vs. generated PAN)
      Generator loss: adversarial (BCE) + SSIM loss + spatial (Sobel) loss
4. Evaluate on the test set (RMSE, SAM, ERGAS, PSNR-B, Sobel correlation).
5. (Optional) Run inference on full-size Sentinel-2 GeoTIFFs with
   overlapping tiles and save georeferenced results.

Conventions
-----------
- NumPy arrays are stored as (N, H, W, C).
- PyTorch tensors are used as (N, C, H, W).

Usage
-----
    python pan_g_ssim_gan.py

Edit the CONFIG section below to match your own paths and hyper-parameters.
"""

import os
import glob
import time
import warnings
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
import sewar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.ndimage import generic_filter, sobel

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE = 128                       # patch size (H = W)
WORK_DIR = "C:/Users/6946/Desktop/code/pan_g"   # working directory (data root)
SAVE_DIR = "pan_g"                      # where checkpoints are stored
BEST_CKPT = os.path.join(SAVE_DIR, "best_generator.pth")

# Landsat training scenes (bands 0-3: MS, band 4: PAN)
LANDSAT_FILES = [
    "landsat/landsat_LQ1_C_31TDL_2018-10-20_10m_b2b3b4b8.tif",
    "landsat/landsat_SR_ARM_C_V2_14SPF_2020-10-11_patches.tif",
    "landsat/landsat_SR_BENGA_LC08_139044_20191121__1.tif",
    "landsat/landsat_SR_BENGA_LC08_139044_20191121__2.tif",
    "landsat/landsat_SR_JAM2018_LC08_220075_20200420__.tif",
    "landsat/landsat_SR_KUDALIAR_LC08_144047_20200416_.tif",
    "landsat/landsat_rapideye_area.tif",
    "landsat/landsat_SR_SO1_LC08_199030_20191008__.tif",
    "landsat/landsat_SR_SO2_LC08_198030_20200222__.tif",
    "landsat/landsat_SR_SUDOUE-2_LC08_199030_20200722.tif",
    "landsat/landsat_SR_SUDOUE-3_LC08_199030_20200908.tif",
    "landsat/landsat_ESGISB-1_30TXQ_2019-02-13_patches.tif",
]

# Inference on Sentinel-2 scenes
RUN_INFERENCE = False                   # set True to process Sentinel-2 tiles
INPUT_DIR = r"C:\Users\6946\Desktop\code\sentinel"
OUTPUT_DIR = r"C:\Users\6946\Desktop\code\sentinel2_5m"
MARGIN = 16                             # border (px) discarded from each patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data helper functions (framework independent)
# =============================================================================
def ergas(img_fake, img_real, scale=2):
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthese).

    Lower is better. Works for 2-D (single band) and 3-D (H, W, C) images.
    """
    if img_fake.shape != img_real.shape:
        raise ValueError("Input images must have the same dimensions.")

    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    eps = np.finfo(np.float64).eps

    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_) ** 2)
        return 100 / scale * np.sqrt(mse / (mean_real ** 2 + eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_) ** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real ** 2 + eps)).mean())
    else:
        raise ValueError("Wrong input image dimensions.")


def image_clip_to_segment(image_ms_array, train_image_array, image_height_size,
                          image_width_size, percentage_overlap, buffer):
    """
    Cut a large image pair into fixed-size patches.

    Parameters
    ----------
    image_ms_array    : array (H, W, C_in)  - network input (here: PAN band used as 'X' arg)
    train_image_array : array (H, W, C_out) - paired image
    image_height_size, image_width_size : patch size
    percentage_overlap : overlap between consecutive patches (0..1)
    buffer             : controls how many border windows are kept

    Returns
    -------
    (train_segment_array, image_segment_array)
        Patches of `train_image_array` and `image_ms_array` respectively.
    """
    # Pad the image so its size is a multiple of the patch size
    y_size = ((image_ms_array.shape[0] // image_height_size) + 1) * image_height_size
    y_pad = int(y_size - image_ms_array.shape[0])
    x_size = ((image_ms_array.shape[1] // image_width_size) + 1) * image_width_size
    x_pad = int(x_size - image_ms_array.shape[1])

    img_complete = np.pad(image_ms_array, ((0, y_pad), (0, x_pad), (0, 0)),
                          mode="symmetric").astype(image_ms_array.dtype)
    train_complete = np.pad(train_image_array, ((0, y_pad), (0, x_pad), (0, 0)),
                            mode="symmetric").astype(train_image_array.dtype)

    img_list, train_list = [], []
    step_i = int((1 - percentage_overlap) * image_height_size)
    step_j = int((1 - percentage_overlap) * image_width_size)

    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), step_i):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), step_j):
            img_list.append(img_complete[i:i + image_height_size, j:j + image_width_size,
                                         0:image_ms_array.shape[2]])
            train_list.append(train_complete[i:i + image_height_size, j:j + image_width_size, :])

    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size,
                                    image_ms_array.shape[2]))
    train_segment_array = np.zeros((len(train_list), image_height_size, image_width_size,
                                    train_image_array.shape[2]))
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        train_segment_array[index] = train_list[index]
    return train_segment_array, image_segment_array


def replace_nan_with_zero(matrix):
    """Replace NaN values with 0."""
    return np.nan_to_num(matrix, nan=0)


def process_landsat_image(filename):
    """
    Read a Landsat GeoTIFF and split it into PAN and MS parts.

    Returns
    -------
    PAN : (H, W, 1)  panchromatic band (band index 4)
    L   : (H, W, 4)  multispectral bands (band indices 0-3)
    """
    with rasterio.open(filename) as landsat:
        data = landsat.read()
    data = np.moveaxis(data, 0, -1)                       # (bands,H,W) -> (H,W,bands)
    PAN = replace_nan_with_zero(data[:, :, 4])[:, :, np.newaxis]
    L = replace_nan_with_zero(data[:, :, 0:4])
    return PAN, L


def to_nchw(a):
    """Convert numpy (N,H,W,C) -> torch float tensor (N,C,H,W)."""
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).float()


def build_datasets(landsat_files, image_size):
    """
    Read all Landsat scenes, cut them into patches and create the
    train / test split (80/20 per scene, then concatenated).
    """
    satellite_data = {f: process_landsat_image(f) for f in landsat_files}

    split_data = []
    for filename, (PAN, L) in satellite_data.items():
        # X: MS (4 bands), y: PAN (1 band)
        X, y = image_clip_to_segment(PAN, L, image_size, image_size, 0, 1)
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=1)
        split_data.append({"train_x": train_x, "train_y": train_y,
                           "test_x": test_x, "test_y": test_y})

    Train_x = np.concatenate([e["train_x"] for e in split_data], axis=0)
    Train_y = np.concatenate([e["train_y"] for e in split_data], axis=0)
    Test_x = np.concatenate([e["test_x"] for e in split_data], axis=0)
    Test_y = np.concatenate([e["test_y"] for e in split_data], axis=0)

    print("Train_x:", Train_x.shape)
    print("Train_y:", Train_y.shape)
    print("Test_x:", Test_x.shape)
    print("Test_y:", Test_y.shape)
    return Train_x, Train_y, Test_x, Test_y


# =============================================================================
# Loss functions
# =============================================================================
def _gaussian_kernel(size=11, sigma=1.5, channels=1):
    """2-D Gaussian kernel shaped for grouped convolution."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_torch(x, y, max_val=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    Structural Similarity Index (equivalent to tf.image.ssim).

    Inputs are (N, C, H, W). Returns one SSIM value per image.
    """
    C = x.shape[1]
    w = _gaussian_kernel(size, sigma, C).to(device=x.device, dtype=x.dtype)

    mu_x = F.conv2d(x, w, groups=C)
    mu_y = F.conv2d(y, w, groups=C)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    s_xx = F.conv2d(x * x, w, groups=C) - mu_xx
    s_yy = F.conv2d(y * y, w, groups=C) - mu_yy
    s_xy = F.conv2d(x * y, w, groups=C) - mu_xy

    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    luminance = (2 * mu_xy + c1) / (mu_xx + mu_yy + c1)
    contrast_structure = (2 * s_xy + c2) / (s_xx + s_yy + c2)
    return (luminance * contrast_structure).mean(dim=(1, 2, 3))


def ssim_loss(y_true, y_pred):
    """SSIM loss = 1 - mean(SSIM)."""
    return 1 - ssim_torch(y_true, y_pred, 1.0).mean()


# Sobel kernels used by the spatial (edge-correlation) loss
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def _corr(a, b, eps=1e-8):
    """Pearson correlation coefficient between two tensors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum()) + eps)


def spatial_loss(y_true, y_pred):
    """
    Spatial loss: encourages edges of the generated image to correlate with
    the edges of the real image (Sobel gradients in x and y direction).
    """
    sx = _SOBEL_X.to(y_true.device, y_true.dtype)
    sy = _SOBEL_Y.to(y_true.device, y_true.dtype)

    true_x, true_y = F.conv2d(y_true, sx, padding=1), F.conv2d(y_true, sy, padding=1)
    pred_x, pred_y = F.conv2d(y_pred, sx, padding=1), F.conv2d(y_pred, sy, padding=1)

    correlation_x = _corr(true_x, pred_x)
    correlation_y = _corr(true_y, pred_y)
    epsilon = 1e-8
    return 1.0 / (correlation_x + correlation_y + epsilon)


# BCE with logits == Keras BinaryCrossentropy(from_logits=True)
bce_logits = nn.BCEWithLogitsLoss()


def generator_loss(generated_output, real_y, generated_images, K0=1, K1=1, K2=0.01):
    """
    Total generator loss = K0 * adversarial + K1 * SSIM + K2 * spatial.
    """
    adv_loss = bce_logits(generated_output, torch.ones_like(generated_output))
    ssim_loss_value = ssim_loss(real_y, generated_images)
    spatial = spatial_loss(real_y, generated_images)
    return K0 * adv_loss + K1 * ssim_loss_value + K2 * spatial


def discriminator_loss(real_output, fake_output):
    """Discriminator loss: real -> 1, fake -> 0."""
    real_loss = bce_logits(real_output, torch.ones_like(real_output))
    fake_loss = bce_logits(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def psnr(target, prediction, max_val=1.0):
    """Peak signal-to-noise ratio (dB) between two tensors."""
    mse = F.mse_loss(prediction, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


# =============================================================================
# Models
# =============================================================================
class ResBlock(nn.Module):
    """Residual block: x + Conv(ReLU(Conv(x)))."""

    def __init__(self, filters, scaling=None):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.scaling = scaling

    def forward(self, x_in):
        x = self.conv2(F.relu(self.conv1(x_in)))
        if self.scaling:
            x = x * self.scaling
        return x_in + x


class Generator(nn.Module):
    """
    Residual generator: MS image (4 bands) -> PAN image (1 band).
    Spatial size is preserved (no up/down sampling).
    """

    def __init__(self, in_channels=4, num_filters=64, num_res_blocks=8,
                 res_block_scaling=None):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_filters, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(num_filters, res_block_scaling)
                                      for _ in range(num_res_blocks)])
        self.body = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.tail = nn.Conv2d(num_filters, 1, 3, padding=1)   # single-band output

    def forward(self, x_in):
        x = self.head(x_in)
        b = self.body(self.blocks(x))
        x = x + b                       # global skip connection
        return self.tail(x)


class Discriminator(nn.Module):
    """CNN classifier: outputs a logit (real vs. generated PAN)."""

    def __init__(self, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )
        feat = img_size // 16           # 4 max-pool layers -> /16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat * feat, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1),           # logit (sigmoid is inside BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def init_weights(m):
    """Glorot-uniform weights and zero bias (Keras default initialisation)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# =============================================================================
# Training
# =============================================================================
def train_step(generator, discriminator, gen_opt, disc_opt, LMS, real_y):
    """
    One GAN training step.

    1) Update the discriminator on real and (detached) fake images.
    2) Update the generator using the adversarial + SSIM + spatial loss.

    Returns (generator_loss, discriminator_loss, psnr) as Python floats.
    """
    generator.train()
    discriminator.train()

    generated_images = generator(LMS)

    # ---- 1) Discriminator update ----
    real_output = discriminator(real_y)
    fake_output = discriminator(generated_images.detach())
    disc_loss = discriminator_loss(real_output, fake_output)

    disc_opt.zero_grad(set_to_none=True)
    disc_loss.backward()
    disc_opt.step()

    # ---- 2) Generator update ----
    generated_output = discriminator(generated_images)
    gen_loss = generator_loss(generated_output, real_y, generated_images)

    gen_opt.zero_grad(set_to_none=True)
    gen_loss.backward()
    gen_opt.step()

    with torch.no_grad():
        psnr_value = psnr(real_y, generated_images.detach())

    return gen_loss.item(), disc_loss.item(), psnr_value.item()


def plot_loss_and_psnr(d_losses, g_losses, psnr_vals):
    """Plot discriminator/generator loss and PSNR over epochs."""
    epochs = range(1, len(d_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, d_losses, label="Discriminator Loss")
    plt.plot(epochs, g_losses, label="Generator Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.title("Discriminator and Generator Losses")

    plt.subplot(2, 1, 2)
    plt.plot(epochs, psnr_vals, label="PSNR", color="orange")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)"); plt.legend(); plt.grid(True)
    plt.title("PSNR over Epochs")

    plt.tight_layout()
    plt.show()


def train(generator, discriminator, train_loader, epochs):
    """
    Full training loop. Saves the generator weights each time the average
    epoch PSNR improves.
    """
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4,
                               betas=(0.9, 0.999), eps=1e-7)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4,
                                betas=(0.9, 0.999), eps=1e-7)

    d_losses_epochs, g_losses_epochs, psnr_epochs = [], [], []
    n_batch = len(train_loader)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_psnr = -1

    for e in range(epochs):
        start_time = time.time()
        gen_loss_total = disc_loss_total = psnr_total = 0.0

        for input_image, target in train_loader:
            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            g, d, p = train_step(generator, discriminator, gen_opt, disc_opt,
                                 input_image, target)
            gen_loss_total += g
            disc_loss_total += d
            psnr_total += p

        avg_gen_loss = gen_loss_total / n_batch
        avg_disc_loss = disc_loss_total / n_batch
        avg_psnr = psnr_total / n_batch

        d_losses_epochs.append(avg_disc_loss)
        g_losses_epochs.append(avg_gen_loss)
        psnr_epochs.append(avg_psnr)

        # Save the best generator (highest PSNR)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), BEST_CKPT)
            print(f"New best PSNR: {best_psnr:.2f} dB. Weights saved.")

        print(f"Epoch {e + 1}: gen loss: {avg_gen_loss:.6f}, "
              f"disc loss: {avg_disc_loss:.6f}, PSNR: {avg_psnr:.2f} dB, "
              f"time: {time.time() - start_time:.2f}s")

    print(f"Training complete. Best generator weights saved in: {BEST_CKPT}")
    plot_loss_and_psnr(d_losses_epochs, g_losses_epochs, psnr_epochs)


# =============================================================================
# Inference & evaluation
# =============================================================================
@torch.no_grad()
def predict(generator, x_nhwc, bs=32):
    """
    Run the generator on a numpy batch.

    Input  : numpy array (N, H, W, C)
    Output : numpy array (N, H, W, 1)
    """
    generator.eval()
    outs = []
    for i in range(0, len(x_nhwc), bs):
        xb = torch.from_numpy(
            np.ascontiguousarray(x_nhwc[i:i + bs].transpose(0, 3, 1, 2))
        ).float().to(device)
        outs.append(generator(xb).cpu().numpy())
    return np.concatenate(outs, axis=0).transpose(0, 2, 3, 1)


def sobel_filter(P):
    """Sobel-like edge magnitude for a flattened 3x3 neighbourhood."""
    return (np.abs((P[0] + 2 * P[1] + P[2]) - (P[6] + 2 * P[7] + P[8])) +
            np.abs((P[2] + 2 * P[6] + P[7]) - (P[0] + 2 * P[3] + P[6])))


def sCC(ms, ps):
    """Spatial correlation coefficient (SCC) based on Sobel edges."""
    ps_sobel = sobel(ps, mode="constant")
    ms_sobel = sobel(ms, mode="constant")
    return (np.sum(ps_sobel * ms_sobel) /
            np.sqrt(np.sum(ps_sobel * ps_sobel)) /
            np.sqrt(np.sum(ms_sobel * ms_sobel)))


def evaluate(Test_y, pred):
    """Print RMSE, SAM, ERGAS, PSNR-B and Sobel correlation metrics."""
    n = Test_y.shape[0]
    print("*****************************************************")

    RMSE = np.mean([sqrt(mean_squared_error(Test_y[i, :, :, 0], pred[i, :, :, 0]))
                    for i in range(n)])
    print("RMSE", round(RMSE, 3))

    SAM = np.mean([sewar.full_ref.sam(Test_y[i].astype(float), pred[i].astype(float))
                   for i in range(n)])
    print("SAM :", round(SAM, 3))

    Ergas = np.mean([ergas(Test_y[i].astype(float), pred[i].astype(float), scale=2)
                     for i in range(n)])
    print("Ergas", round(Ergas, 3))

    PSNR = np.mean([sewar.full_ref.psnrb(Test_y[i].astype(float), pred[i].astype(float))
                    for i in range(n)])
    print("PSNR", round(PSNR, 3))

    sobel_pan_ = np.concatenate([generic_filter(Test_y[i, :, :, 0], sobel_filter, (3, 3)).ravel()
                                 for i in range(n)])
    sobel_G_ = np.concatenate([generic_filter(pred[i, :, :, 0], sobel_filter, (3, 3)).ravel()
                               for i in range(n)])
    r_sobel = np.corrcoef(sobel_pan_, sobel_G_)
    print("r coef_sobel:", round(r_sobel[1, 0], 2))

    sobel_band0_ = np.mean([sCC(Test_y[i, :, :, 0], pred[i, :, :, 0]) for i in range(n)])
    print("r coef_sobel band0:", round(sobel_band0_, 3))
    print("*****************************************************")


def show_sample(Test_x, Test_y, pred, n=10):
    """Show the predicted PAN, the MS composite and the real PAN of sample n."""
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Predict_images")
    plt.axis("Off")
    ax.imshow(pred[n, :, :, 0], cmap="gray")

    ax = fig.add_subplot(1, 3, 2)
    MS = np.dstack((Test_x[n, :, :, 1], Test_x[n, :, :, 2], Test_x[n, :, :, 3]))
    MS = np.clip(MS / np.amax(MS), 0, 1)
    ax.set_title("MultiSpectral")
    plt.axis("Off")
    ax.imshow(MS)

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(Test_y[n, :, :, 0], cmap="gray")
    ax.set_title("PAN")
    plt.axis("Off")

    plt.tight_layout()
    plt.show()


def process_sentinel_tif(image_path, generator, save_path, patch=IMAGE_SIZE,
                         margin=MARGIN):
    """
    Apply the trained generator to a full-size Sentinel-2 GeoTIFF.

    The image is processed in overlapping patches. Only the central part of
    each patch prediction is kept (a `margin` border is discarded) to avoid
    visible seams. The output GeoTIFF keeps the georeferencing of the input;
    if the generator changes the resolution, the affine transform is rescaled.
    """
    try:
        # 1) Read image as (bands, H, W) keeping the file band order
        with rasterio.open(image_path) as src:
            img = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

        bands, H, W = img.shape
        core = patch - 2 * margin               # valid central size of each patch

        n_rows = int(np.ceil(H / core))
        n_cols = int(np.ceil(W / core))

        # 2) Reflect-pad: `margin` on every side + extra to complete the grid
        extra_h = n_rows * core - H
        extra_w = n_cols * core - W
        padded = np.pad(img,
                        ((0, 0), (margin, margin + extra_h), (margin, margin + extra_w)),
                        mode="reflect")

        out = None
        f = None  # scale factor of output vs. input (1 or 2 ...)

        # 3) Process one row of patches at a time
        for i in range(n_rows):
            r = i * core
            row_patches = []
            for j in range(n_cols):
                c = j * core
                p = padded[:, r:r + patch, c:c + patch]            # (C, 128, 128)
                row_patches.append(np.transpose(p, (1, 2, 0)))     # (128, 128, C)
            batch = np.stack(row_patches, axis=0)                  # (n_cols, 128, 128, C)

            pred_row = predict(generator, batch)[..., 0]           # (n_cols, h, w)

            if out is None:
                f = pred_row.shape[1] / patch
                m_out = int(margin * f)
                core_out = int(core * f)
                out = np.zeros((n_rows * core_out, n_cols * core_out), dtype=np.float32)

            # 4) Keep only the central part of each patch
            for j in range(n_cols):
                center = pred_row[j, m_out:m_out + core_out, m_out:m_out + core_out]
                rr = i * core_out
                cc = j * core_out
                out[rr:rr + core_out, cc:cc + core_out] = center

        # 5) Crop the extra padding area
        out = out[:int(H * f), :int(W * f)]

        # 6) Adjust the transform if pixel size changed
        new_transform = transform * Affine.scale(1 / f, 1 / f)

        profile.update(driver="GTiff", count=1, dtype="float32",
                       height=out.shape[0], width=out.shape[1],
                       transform=new_transform, crs=crs, compress="lzw")

        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(out, 1)

        print(f"Processed and saved: {save_path}  ({out.shape[1]}x{out.shape[0]})")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def run_sentinel_inference(generator):
    """Process every .tif in INPUT_DIR and write the results to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    print(f"{len(tif_files)} files found")

    for image_path in tif_files:
        name = os.path.basename(image_path)
        save_path = os.path.join(OUTPUT_DIR, "PAN_" + name)
        process_sentinel_tif(image_path, generator, save_path)


# =============================================================================
# Main
# =============================================================================
def main():
    print("Device:", device, "| GPUs available:", torch.cuda.device_count())
    os.chdir(WORK_DIR)

    # ---- Data ----
    Train_x, Train_y, Test_x, Test_y = build_datasets(LANDSAT_FILES, IMAGE_SIZE)

    train_ds = TensorDataset(to_nchw(Train_x), to_nchw(Train_y))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # ---- Models ----
    generator = Generator().to(device)
    discriminator = Discriminator(IMAGE_SIZE).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    print("Generator params    :", sum(p.numel() for p in generator.parameters()))
    print("Discriminator params:", sum(p.numel() for p in discriminator.parameters()))

    # ---- Training ----
    train(generator, discriminator, train_loader, EPOCHS)

    # ---- Load best weights ----
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    generator.eval()

    # ---- Evaluation on the test set ----
    pred = predict(generator, Test_x)
    print(pred.shape)
    evaluate(Test_y, pred)
    show_sample(Test_x, Test_y, pred, n=10)

    # ---- Optional: inference on Sentinel-2 scenes ----
    if RUN_INFERENCE:
        run_sentinel_inference(generator)


if __name__ == "__main__":
    main()
