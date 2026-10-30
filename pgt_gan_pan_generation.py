
"""
PGT-GAN: PAN-Generation Transformer GAN for Sentinel-2 spatial detail enhancement
=================================================================================

PyTorch implementation following the paper
"Synthetic Panchromatic Band Generation Using a Transformer-GAN for Spatial
Detail Enhancement of Sentinel-2 Images".

Idea
----
Sentinel-2 has no panchromatic (PAN) band, so classical pansharpening cannot be
applied directly. A GAN is trained on Landsat-8 (which has a real PAN band) to
map a 4-band (R, G, B, NIR) low-resolution multispectral (LRMS) patch of size
HxW to a single-band PAN patch of size 2Hx2W (scale factor 2). The trained
generator is then applied to Sentinel-2 RGBN images to obtain a synthetic
PAN band at a target 5-m spatial-detail level. This synthetic PAN can be fused
with the Sentinel-2 MS bands with any pansharpening method (BDSD, HPF, ...),
which is done outside of this script.

Model
-----
Generator (hierarchical Transformer encoder-decoder, 3 levels, 1 block/level)
    input  : (N, 4, H,  W )   LRMS
    output : (N, 1, 2H, 2W)   PAN in [0, 1] (sigmoid)
    Transformer block = Multi-Dconv-head Transposed Attention (MDTA)
                      + Gated-Dconv Feed-Forward Network (GDFN)
Discriminator
    3 x [Conv (7x7 / 5x5 / 3x3, 32 / 64 / 128 filters) + LeakyReLU + MaxPool]
    -> Flatten -> FC128 -> FC16 -> FC1 (sigmoid probability)

Generator loss (Eq. 10 in the paper)
    L_G = a1 * L_adv + a2 * L_SSIM + a3 * L_spatial      (a1, a2, a3 = 1, 1, 0.1)

Training set-up (Section 4.3 of the paper)
    PyTorch, batch size 32, 100 epochs, Adam (beta1=0.9, beta2=0.999),
    learning rate 1e-5 (generator) and 1e-4 (discriminator).

Conventions
-----------
- NumPy arrays are stored as (N, H, W, C).
- PyTorch tensors are used as (N, C, H, W).

Usage
-----
    python pgt_gan_pan_generation.py
Edit the CONFIG section below to match your paths.
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
# ---- training hyper-parameters (paper, Section 4.3) ----
BATCH_SIZE = 32
EPOCHS = 100
LR_G = 1e-5                 # generator learning rate
LR_D = 1e-4                 # discriminator learning rate
LOSS_A1, LOSS_A2, LOSS_A3 = 1.0, 1.0, 0.1   # adversarial / SSIM / spatial weights

# ---- patch geometry ----
SCALE = 2                   # PAN is generated at 2x the MS size
HR_SIZE = 128               # PAN (target) patch size
LR_SIZE = HR_SIZE // SCALE  # MS (input) patch size -> 64

# ---- model ----
BASE_FILTERS = 48           # channels C of the first encoder level
                            # (must be divisible by 2; heads per level = 1, 2, 4)

# ---- paths ----
WORK_DIR = "C:/Users/6946/Desktop/code/pan_g"   # working directory (data root)
SAVE_DIR = "pan_g"                               # checkpoint directory
BEST_CKPT = os.path.join(SAVE_DIR, "best_generator_pgt.pth")

# Landsat-8 training scenes (bands 0-3: MS (R,G,B,NIR order), band 4: PAN).
# Reflectance values are expected to be in [0, 1].
LANDSAT_FILES = [
    "landsat/landsat_LQ1_C_31TDL_2018-10-20_10m_b2b3b4b8.tif",
    "landsat/landsat_SR_ARM_C_V2_14SPF_2020-10-11_patches.tif",
    "landsat/landsat_SR_BENGA_LC08_139044_20191121__1.tif",
    "landsat/landsat_SR_BENGA_LC08_139044_20191121__2.tif",
    "landsat/landsat_SR_JAM2018_LC08_220075_20200420__.tif",
    "landsat/landsat_SR_KUDALIAR_LC08_144047_20200416_.tif",
    "landsat/landsat_SR_SO1_LC08_199030_20191008__.tif",
    "landsat/landsat_SR_SO2_LC08_198030_20200222__.tif",
    "landsat/landsat_SR_SUDOUE-2_LC08_199030_20200722.tif",
    "landsat/landsat_SR_SUDOUE-3_LC08_199030_20200908.tif",
    "landsat/landsat_ESGISB-1_30TXQ_2019-02-13_patches.tif",
]

# ---- inference on Sentinel-2 scenes ----
RUN_INFERENCE = False       # set True to generate PAN for Sentinel-2 GeoTIFFs
INPUT_DIR = r"C:\Users\6946\Desktop\code\sentinel"
OUTPUT_DIR = r"C:\Users\6946\Desktop\code\sentinel2_5m"
INFER_PATCH = LR_SIZE       # input patch size used at inference (same as training)
MARGIN = 16                 # border (px) discarded from each predicted patch
SENTINEL_SCALE = 1.0        # divide Sentinel-2 values by this (use 10000 for raw DN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data helper functions
# =============================================================================
def ergas(img_fake, img_real, scale=2):
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthese). Lower is better.
    Works for 2-D (single band) and 3-D (H, W, C) images.
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
    image_ms_array    : (H, W, C1) first image  (here: the PAN band)
    train_image_array : (H, W, C2) second image (here: the MS bands)
    image_height_size, image_width_size : patch size
    percentage_overlap : overlap between consecutive patches (0..1)
    buffer             : controls how many border windows are kept

    Returns
    -------
    (train_segment_array, image_segment_array) = (patches of 2nd image, patches of 1st image)
    """
    # Pad so the image size is a multiple of the patch size
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
                                    image_ms_array.shape[2]), dtype=np.float32)
    train_segment_array = np.zeros((len(train_list), image_height_size, image_width_size,
                                    train_image_array.shape[2]), dtype=np.float32)
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        train_segment_array[index] = train_list[index]
    return train_segment_array, image_segment_array


def process_landsat_image(filename):
    """
    Read a Landsat-8 GeoTIFF and split it into PAN and MS parts.

    NaNs are replaced by 0 and values are clipped to [0, 1] because the generator
    ends with a sigmoid (outputs in [0, 1]).

    Returns
    -------
    PAN : (H, W, 1)  panchromatic band (band index 4)
    L   : (H, W, 4)  multispectral bands (band indices 0-3)
    """
    with rasterio.open(filename) as landsat:
        data = landsat.read()
    data = np.moveaxis(data, 0, -1)                        # (bands,H,W) -> (H,W,bands)
    data = np.clip(np.nan_to_num(data, nan=0), 0, 1)
    PAN = data[:, :, 4:5]
    L = data[:, :, 0:4]
    return PAN, L


def downsample_mean(a, s):
    """
    Average-pool a (N, H, W, C) numpy array by an integer factor `s`.
    Used to obtain the LRMS input (H/s x W/s) from an MS patch that lives on the
    PAN grid.
    """
    n, h, w, c = a.shape
    return a.reshape(n, h // s, s, w // s, s, c).mean(axis=(2, 4))


def to_nchw(a):
    """Convert numpy (N,H,W,C) -> torch float tensor (N,C,H,W)."""
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).float()


def build_datasets(landsat_files, hr_size, scale):
    """
    Build the training / test arrays.

    For every Landsat scene:
      * cut HR patches (hr_size x hr_size) of PAN (target) and MS,
      * average-pool the MS patch by `scale` -> LRMS input (hr_size/scale),
      * split patches 80 / 20 into train / test.

    Returns
    -------
    Train_x (N, hr/s, hr/s, 4), Train_y (N, hr, hr, 1), Test_x, Test_y
    """
    split_data = []
    for filename in landsat_files:
        PAN, L = process_landsat_image(filename)
        # X: MS patches (hr x hr), y: PAN patches (hr x hr)
        X, y = image_clip_to_segment(PAN, L, hr_size, hr_size, 0, 1)
        X = downsample_mean(X, scale).astype(np.float32)       # LRMS: (hr/s x hr/s)
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=1)
        split_data.append({"train_x": train_x, "train_y": train_y,
                           "test_x": test_x, "test_y": test_y})

    Train_x = np.concatenate([e["train_x"] for e in split_data], axis=0)
    Train_y = np.concatenate([e["train_y"] for e in split_data], axis=0)
    Test_x = np.concatenate([e["test_x"] for e in split_data], axis=0)
    Test_y = np.concatenate([e["test_y"] for e in split_data], axis=0)

    print("Train_x (LRMS):", Train_x.shape)
    print("Train_y (PAN) :", Train_y.shape)
    print("Test_x  (LRMS):", Test_x.shape)
    print("Test_y  (PAN) :", Test_y.shape)
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
    Structural Similarity Index (Eq. 12 of the paper; equivalent to tf.image.ssim).
    C1 = (K1*L)^2, C2 = (K2*L)^2 with K1 = 0.01, K2 = 0.03.
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
    """L_SSIM = 1 - SSIM(P_G, P_R)."""
    return 1 - ssim_torch(y_true, y_pred, 1.0).mean()


# Sobel kernels for the spatial (edge-correlation) loss
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
    L_spatial = 1 / (rho_x + rho_y + eps)   (Eq. 13 of the paper)
    where rho_x / rho_y are the correlations between the horizontal / vertical
    Sobel edges of the reference and generated PAN images.
    """
    sx = _SOBEL_X.to(y_true.device, y_true.dtype)
    sy = _SOBEL_Y.to(y_true.device, y_true.dtype)

    true_x, true_y = F.conv2d(y_true, sx, padding=1), F.conv2d(y_true, sy, padding=1)
    pred_x, pred_y = F.conv2d(y_pred, sx, padding=1), F.conv2d(y_pred, sy, padding=1)

    rho_x = _corr(true_x, pred_x)
    rho_y = _corr(true_y, pred_y)
    epsilon = 1e-8
    return 1.0 / (rho_x + rho_y + epsilon)


# The discriminator outputs a logit; the sigmoid of the paper is applied inside
# BCEWithLogitsLoss (numerically more stable, mathematically identical).
bce_logits = nn.BCEWithLogitsLoss()


def generator_loss(generated_output, real_y, generated_images,
                   a1=LOSS_A1, a2=LOSS_A2, a3=LOSS_A3):
    """L_G = a1 * L_adv + a2 * L_SSIM + a3 * L_spatial   (Eq. 10)."""
    adv_loss = bce_logits(generated_output, torch.ones_like(generated_output))
    return (a1 * adv_loss +
            a2 * ssim_loss(real_y, generated_images) +
            a3 * spatial_loss(real_y, generated_images))


def discriminator_loss(real_output, fake_output):
    """Discriminator loss: real -> 1, generated -> 0   (Eq. 11b)."""
    real_loss = bce_logits(real_output, torch.ones_like(real_output))
    fake_loss = bce_logits(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def psnr(target, prediction, max_val=1.0):
    """Peak signal-to-noise ratio (dB) between two tensors."""
    mse = F.mse_loss(prediction, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


# =============================================================================
# Transformer building blocks (Restormer-style, as used in SRTransGAN)
# =============================================================================
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """
    Multi-Dconv-head Transposed Attention.

    Q, K, V = depth-wise 3x3 conv( point-wise 1x1 conv (Y) ).
    Attention is computed across channels (a C x C map) instead of across pixels,
    so the cost grows linearly with the image size.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)                       # W_p
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                groups=dim * 3, bias=bias)                      # W_d
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)

        # reshape to (B, heads, C/heads, H*W)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature                     # (B, heads, C', C')
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (Eq. 9).
    Two parallel 1x1 -> depth-wise 3x3 paths; one passes through GELU and gates
    the other by element-wise multiplication.
    """

    def __init__(self, dim, expansion=2, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """LN -> MDTA (+skip) -> LN -> GDFN (+skip)."""

    def __init__(self, dim, num_heads, expansion=2):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """Halve spatial size and double channels (conv + PixelUnshuffle)."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial size and halve channels (conv + PixelShuffle)."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# =============================================================================
# Generator / Discriminator
# =============================================================================
class Generator(nn.Module):
    """
    PGT-GAN generator: hierarchical Transformer encoder-decoder (l = 3 levels,
    T = 1 transformer block per level).

    Input  : (N, in_channels, H,  W )  LRMS
    Output : (N, 1,           2H, 2W)  synthetic PAN in [0, 1]

    Encoder (resolution / channels)
        F0 = Conv3x3(X)                       H   x C
        F1 = E1(F0)                           H   x C
        F2 = E2(Down(F1))                     H/2 x 2C
        F3 = E3(Down(F2))                     H/4 x 4C
        Fb = Conv3x3(ReLU(Conv3x3(F3)))       bottleneck (Eq. 4)
    Decoder (each step: Up -> concat skip -> 1x1 conv to reduce channels -> block)
        F^2 = D3(reduce(Up(Fb)  (+) F2))      H/2 x 2C
        F^1 = D2(reduce(Up(F^2) (+) F1))      H   x C
        F^0 = D1(reduce(Up(F^1) (+) up(F0)))  2H  x C/2   (skip is bilinearly
                                                           up-sampled to 2H)
    Output (Eq. 8): sigmoid(Conv3x3(ReLU(Conv3x3(F^0))))
    """

    def __init__(self, in_channels=4, dim=BASE_FILTERS, num_blocks=1, heads=(1, 2, 4)):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"

        # ---- low-level feature extraction ----
        self.head = nn.Conv2d(in_channels, dim, 3, padding=1)

        # ---- encoder ----
        self.enc1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks)])
        self.down1 = Downsample(dim)
        self.enc2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks)])
        self.down2 = Downsample(dim * 2)
        self.enc3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2]) for _ in range(num_blocks)])

        # ---- bottleneck ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim * 4, dim * 4, 3, padding=1))

        # ---- decoder ----
        self.up3 = Upsample(dim * 4)                              # -> 2C @ H/2
        self.reduce3 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)  # cat(2C, 2C) -> 2C
        self.dec3 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks)])

        self.up2 = Upsample(dim * 2)                              # -> C @ H
        self.reduce2 = nn.Conv2d(dim * 2, dim, 1, bias=False)      # cat(C, C) -> C
        self.dec2 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks)])

        self.up1 = Upsample(dim)                                  # -> C/2 @ 2H  (scale x2)
        self.reduce1 = nn.Conv2d(dim // 2 + dim, dim // 2, 1, bias=False)
        self.dec1 = nn.Sequential(*[TransformerBlock(dim // 2, 1) for _ in range(num_blocks)])

        # ---- output ----
        self.out = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        # encoder
        f0 = self.head(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        fb = self.bottleneck(f3)

        # decoder with skip connections
        d2 = self.dec3(self.reduce3(torch.cat([self.up3(fb), f2], dim=1)))
        d1 = self.dec2(self.reduce2(torch.cat([self.up2(d2), f1], dim=1)))
        s0 = F.interpolate(f0, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.dec1(self.reduce1(torch.cat([self.up1(d1), s0], dim=1)))

        return self.out(d0)


class Discriminator(nn.Module):
    """
    Discriminator (Section 3.2): three Conv + LeakyReLU + MaxPool stages with
    32 / 64 / 128 filters and kernels 7x7 / 5x5 / 3x3, followed by FC 128 -> 16 -> 1.
    The last layer returns a logit (the sigmoid is inside BCEWithLogitsLoss).
    """

    def __init__(self, img_size=HR_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool2d(2),
        )
        feat = img_size // 8             # 3 max-pool layers -> /8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat * feat, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 16), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def init_weights(m):
    """Glorot-uniform weights and zero bias for Conv / Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =============================================================================
# Training
# =============================================================================
def train_step(generator, discriminator, gen_opt, disc_opt, LMS, real_y):
    """
    One GAN training step.
      1) update the discriminator on real and (detached) generated PAN images,
      2) update the generator with the composite loss.
    Returns (generator_loss, discriminator_loss, psnr) as Python floats.
    """
    generator.train()
    discriminator.train()

    generated_images = generator(LMS)                        # (N, 1, 2H, 2W)

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
    Full training loop with asymmetric learning rates (G: 1e-5, D: 1e-4).
    The generator is saved every time the average epoch PSNR improves.
    """
    gen_opt = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.9, 0.999))
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.9, 0.999))

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
    Input  : (N, H,  W,  C)
    Output : (N, 2H, 2W, 1)
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
    """Print RMSE, MAE, SSIM, PSNR, SAM, ERGAS and SCC on the test set."""
    n = Test_y.shape[0]
    print("*****************************************************")

    RMSE = np.mean([sqrt(mean_squared_error(Test_y[i, :, :, 0], pred[i, :, :, 0]))
                    for i in range(n)])
    print("RMSE :", round(RMSE, 4))

    MAE = np.mean(np.abs(Test_y - pred))
    print("MAE  :", round(float(MAE), 4))

    SSIM = np.mean([sewar.full_ref.ssim(Test_y[i].astype(float), pred[i].astype(float),
                                        MAX=1.0)[0] for i in range(n)])
    print("SSIM :", round(SSIM, 3))

    PSNR = np.mean([sewar.full_ref.psnrb(Test_y[i].astype(float), pred[i].astype(float))
                    for i in range(n)])
    print("PSNR :", round(PSNR, 3))

    SAM = np.mean([sewar.full_ref.sam(Test_y[i].astype(float), pred[i].astype(float))
                   for i in range(n)])
    print("SAM  :", round(SAM, 3))

    Ergas = np.mean([ergas(Test_y[i].astype(float), pred[i].astype(float), scale=SCALE)
                     for i in range(n)])
    print("ERGAS:", round(Ergas, 3))

    SCC = np.mean([sCC(Test_y[i, :, :, 0], pred[i, :, :, 0]) for i in range(n)])
    print("SCC  :", round(SCC, 3))

    sobel_pan_ = np.concatenate([generic_filter(Test_y[i, :, :, 0], sobel_filter, (3, 3)).ravel()
                                 for i in range(n)])
    sobel_G_ = np.concatenate([generic_filter(pred[i, :, :, 0], sobel_filter, (3, 3)).ravel()
                               for i in range(n)])
    print("r coef_sobel:", round(np.corrcoef(sobel_pan_, sobel_G_)[1, 0], 2))
    print("*****************************************************")


def show_sample(Test_x, Test_y, pred, n=10):
    """Show the generated PAN, the LRMS composite and the reference PAN of sample n."""
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Generated PAN")
    plt.axis("Off")
    ax.imshow(pred[n, :, :, 0], cmap="gray")

    ax = fig.add_subplot(1, 3, 2)
    MS = np.dstack((Test_x[n, :, :, 1], Test_x[n, :, :, 2], Test_x[n, :, :, 3]))
    MS = np.clip(MS / np.amax(MS), 0, 1)
    ax.set_title("MultiSpectral (input)")
    plt.axis("Off")
    ax.imshow(MS)

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(Test_y[n, :, :, 0], cmap="gray")
    ax.set_title("Reference PAN")
    plt.axis("Off")

    plt.tight_layout()
    plt.show()


def process_sentinel_tif(image_path, generator, save_path, patch=INFER_PATCH,
                         margin=MARGIN):
    """
    Generate the synthetic PAN band for a full-size Sentinel-2 GeoTIFF.

    The input must contain the R, G, B, NIR bands (in the same order as the
    Landsat training data) as surface reflectance. The image is processed in
    overlapping patches; only the central part of every prediction is kept
    (a `margin` border is discarded) to avoid seams. The output pixel size is
    half of the input one (scale x2), so the affine transform is rescaled.
    """
    try:
        # 1) Read image as (bands, H, W) and normalise to [0, 1]
        with rasterio.open(image_path) as src:
            img = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
        img = np.clip(np.nan_to_num(img / SENTINEL_SCALE, nan=0), 0, 1)

        bands, H, W = img.shape
        core = patch - 2 * margin                  # valid central size of each patch

        n_rows = int(np.ceil(H / core))
        n_cols = int(np.ceil(W / core))

        # 2) Reflect-pad: `margin` on every side + extra to complete the grid
        extra_h = n_rows * core - H
        extra_w = n_cols * core - W
        padded = np.pad(img,
                        ((0, 0), (margin, margin + extra_h), (margin, margin + extra_w)),
                        mode="reflect")

        out = None
        f = None  # output / input scale factor (2 for this model)

        # 3) Process one row of patches at a time
        for i in range(n_rows):
            r = i * core
            row_patches = []
            for j in range(n_cols):
                c = j * core
                p = padded[:, r:r + patch, c:c + patch]            # (C, patch, patch)
                row_patches.append(np.transpose(p, (1, 2, 0)))     # (patch, patch, C)
            batch = np.stack(row_patches, axis=0)

            pred_row = predict(generator, batch)[..., 0]           # (n_cols, f*patch, f*patch)

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

        # 5) Crop the padded area
        out = out[:int(H * f), :int(W * f)]

        # 6) Rescale the transform (pixel size is divided by f)
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
    """Process every .tif in INPUT_DIR and write the synthetic PAN to OUTPUT_DIR."""
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

    # ---- Data: LRMS (64x64x4) -> PAN (128x128x1) ----
    Train_x, Train_y, Test_x, Test_y = build_datasets(LANDSAT_FILES, HR_SIZE, SCALE)

    train_ds = TensorDataset(to_nchw(Train_x), to_nchw(Train_y))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # ---- Models ----
    generator = Generator().to(device)
    discriminator = Discriminator(HR_SIZE).to(device)
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
    print("Prediction shape:", pred.shape)
    evaluate(Test_y, pred)
    show_sample(Test_x, Test_y, pred, n=10)

    # ---- Optional: synthetic PAN for Sentinel-2 scenes ----
    if RUN_INFERENCE:
        run_sentinel_inference(generator)


if __name__ == "__main__":
    main()
