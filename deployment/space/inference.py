import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

SEMANTIC_CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "lymphocyte",
    3: "neutrophil",
    4: "macrophage",
}

TER_CLASS_NAMES = {
    0: "background",
    1: "inside",
    2: "boundary",
}

SEM_COLORS = {
    0: None,                # background - no overlay
    1: (255, 0, 0),         # epithelial - red
    2: (0, 128, 255),       # lymphocyte - blue
    3: (255, 255, 0),       # neutrophil - yellow
    4: (0, 255, 0),         # macrophage - green
}

TER_COLORS = {
    0: None,                # background - no overlay
    1: (0, 200, 100),       # inside - green
    2: (255, 100, 0),       # boundary - orange
}

TILE_SIZE = 256
OVERLAP = 64
STRIDE = TILE_SIZE - OVERLAP    # = 192

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    # convert to RGB
    img = pil_image.convert("RGB")
    # convert to float32 in rnage [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0      # [H, W, 3] in [0, 1]
    # applies ImageNet normalization (pixel - mean) / std
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    # return numpy array of shape [H, W, 3]
    return img_np

def run_tiled_inference(model, img_np, device="cpu"):
    H, W = img_np.shape[:2]
    pad_h = (STRIDE - H % STRIDE) % STRIDE
    pad_w = (STRIDE - W % STRIDE) % STRIDE

    pad_h = max(pad_h, TILE_SIZE - H) if H < TILE_SIZE else pad_h
    pad_w = max(pad_w, TILE_SIZE - W) if W < TILE_SIZE else pad_w
    padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    def _gaussian_weight(size, sigma=None):
        if sigma is None:
            sigma = size / 4
        ax = np.arange(size) - size / 2 + 0.5
        gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        gauss_2d = np.outer(gauss_1d, gauss_1d)
        return gauss_2d.astype(np.float32)
    
    Hp, Wp = padded.shape[:2]
    coords = [(y, x) for y in range(0, Hp - TILE_SIZE + 1, STRIDE)
                        for x in range(0, Wp - TILE_SIZE + 1, STRIDE)]
    
    sem_accum = np.zeros((5, Hp, Wp), dtype=np.float32)     # 5 semantic classes
    ter_accum = np.zeros((3, Hp, Wp), dtype=np.float32)     # 3 ternary classes
    weight_accum = np.zeros((1, Hp, Wp), dtype=np.float32)
    gauss = _gaussian_weight(TILE_SIZE)

    model = model.to(device).eval().float()

    with torch.no_grad():
        for (y, x) in coords:
            tile = padded[y:y+TILE_SIZE, x:x+TILE_SIZE]         # [256, 256, 3]
            tensor = torch.from_numpy(tile).permute(2, 0, 1)    # [3, 256, 256]
            tensor = tensor.unsqueeze(0).to(device).float()     # [1, 3, 256, 256]
            
            sem_logits, ter_logits = model(tensor)              # [1, 5, 256, 256]

            sem_np = sem_logits[0].cpu().numpy()                # [5, 256, 256]
            ter_np = ter_logits[0].cpu().numpy()                # [3, 256, 256]

            sem_accum[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += sem_np * gauss
            ter_accum[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += ter_np * gauss
            weight_accum[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += gauss

    sem_accum /= (weight_accum + 1e-8)
    ter_accum /= (weight_accum + 1e-8)

    # crop back to original size
    sem_accum = sem_accum[:, :H, :W]
    ter_accum = ter_accum[:, :H, :W]

    # argmax for class predictions
    sem_pred = np.argmax(sem_accum, axis=0)     # [H, W]
    ter_pred = np.argmax(ter_accum, axis=0)     # [H, W]

    return (sem_pred, ter_pred)

def overlay_segmentation(img_rgb, pred_map, color_dict, alpha=0.5):
    # takes original RGB image as [H, W, 3] uint8
    overlay = img_rgb.copy()
    # takes predict map [H, W] (class indices)
    for class_id, color in color_dict.items():
        if color is None:
            continue
        mask = (pred_map == class_id)
        overlay[mask] = (
            alpha * np.array(color, dtype=np.float32)
            + (1 - alpha) * overlay[mask].astype(np.float32)
        ).astype(np.uint8)
    return overlay