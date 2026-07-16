import os
from pathlib import Path

import numpy as np
import torch
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

CAMERAS = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT"
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

rain_aug = K.RandomRainElliptical(
    number_of_drops=(5000, 10000),
    drop_shape="ellipse",
    ellipse_height_mean=20,
    ellipse_height_std=3,
    ellipse_width_mean=2,
    ellipse_width_std=0.1,
    ellipse_angle_range=(-30.0, 30.0),
    rain_color=(0.9, 0.9, 1.0),
    alpha=0.50,
    p=1.0,
)

snow_aug = K.RandomSnowManual(
    snow_coefficient=(0.2, 0.6),
    flake_height_mean=15,
    flake_height_std=5,
    flake_width_mean=3,
    flake_width_std=1,
    alpha=0.6,
    angle_range=(-15, 15),
    p=1.0,
)

augmentor = snow_aug

def process_camera(cam_name, max_img = None):
    input_dir = ROOT_DIR / cam_name
    output_dir = OUTPUT_DIR / cam_name

    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if max_img is not None:
        input_paths = input_paths[:max_img]

    for img_path in tqdm(input_paths, desc=f"Processing {cam_name}"):
        img = Image.open(img_path).convert("RGB")

        img_tensor = kornia.image_to_tensor(
            np.array(img), keepdim=False
        ).float() / 255.0

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            rain_tensor = augmentor(img_tensor)

        rain_tensor = rain_tensor.squeeze(0).clamp(0, 1).cpu()

        output_img_np = (
            kornia.tensor_to_image(rain_tensor) * 255
        ).astype(np.uint8)

        output_img = Image.fromarray(output_img_np)
        save_path = output_dir / img_path.name
        output_img.save(save_path)


def main():
    for cam in CAMERAS:
        print(f"Starting {cam}")
        process_camera(cam)

    print("Done.")


if __name__ == "__main__":
    main()