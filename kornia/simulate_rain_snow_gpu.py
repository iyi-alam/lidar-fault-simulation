import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import kornia
import kornia.augmentation as K
from PIL import Image
from tqdm import tqdm


ROOT_DIR = Path(
    "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/wedit_bg/snow_wedit_bg/samples"
)

OUTPUT_DIR = Path(
    "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/wedit_bg/snow_wedit_particles/samples"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

CAMERAS = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]

BATCH_SIZE = 32
NUM_WORKERS = 0
USE_AMP = False

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_np = np.array(img, copy=True)

        img_tensor = kornia.image_to_tensor(img_np).float() / 255.0

        return img_tensor, img_path.name


def collate_fn(batch):
    tensors = [x[0] for x in batch]
    names = [x[1] for x in batch]
    return torch.stack(tensors), names


snow_aug = K.RandomSnowManual(
    snow_coefficient=(0.5, 0.7),
    flake_height_mean=15,
    flake_height_std=5,
    flake_width_mean=5,
    flake_width_std=3,
    alpha=0.8,
    angle_range=(-15, 15),
    p=1.0,
).to(device)


def save_batch(output_batch, names, output_dir):
    output_batch = output_batch.clamp(0, 1).cpu()

    for tensor, name in zip(output_batch, names):
        img_np = (kornia.tensor_to_image(tensor) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(output_dir / name)


def process_camera(cam_name):
    input_dir = ROOT_DIR / cam_name
    output_dir = OUTPUT_DIR / cam_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    dataset = ImageDataset(image_paths)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    with torch.inference_mode():
        for batch, names in tqdm(dataloader, desc=cam_name):
            batch = batch.to(device)

            output = snow_aug(batch)

            save_batch(output, names, output_dir)


def main():
    print("Using device:", device)

    for cam in CAMERAS:
        print("Starting", cam)
        process_camera(cam)

    print("Done")


if __name__ == "__main__":
    main()