import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)

    @torch.no_grad()
    def forward(self, x):
        # x: [B,C,H,W], values in [0,1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x.clamp(0, 1)

        features = self.model(x)
        return features
    
def get_features_from_loader(loader, feature_extractor, device, max_batches=20):
    features = []

    feature_extractor.eval()

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break

            x = x.to(device)
            feat = feature_extractor(x)
            features.append(feat.detach().cpu())

    return torch.cat(features, dim=0).numpy()

def get_features_from_tensor(images, feature_extractor, device, batch_size=64):
    features = []

    feature_extractor.eval()

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i+batch_size].to(device)
            feat = feature_extractor(x)
            features.append(feat.detach().cpu())

    return torch.cat(features, dim=0).numpy()


def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6):
    diff = mu_real - mu_gen

    covmean = sqrtm(sigma_real @ sigma_gen)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return float(fid)


def get_features_from_loader(loader, feature_extractor, device, max_batches=20):
    features = []

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break

            x = x.to(device)
            feat = feature_extractor(x)
            features.append(feat.detach().cpu())

    return torch.cat(features, dim=0).numpy()

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma