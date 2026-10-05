import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config


FOLDER_CAR_MAP = {
    "001": "Sedan", "002": "Sedan", "003": "Sedan", "004": "Sedan",
    "005": "Sedan", "006": "Sedan", "007": "Sedan", "008": "Sedan",
    "009": "Hatchback", "010": "Hatchback", "011": "Coupe", "012": "Pickup Truck",
    "013": "Hatchback", "014": "SUV", "015": "Pickup Truck", "016": "Sedan",
    "017": "SUV", "018": "SUV", "019": "Van / Minivan", "020": "Sedan",
    "021": "Hatchback", "022": "SUV", "023": "Sedan", "024": "Sedan",
    "025": "SUV", "026": "Van / Minivan", "027": "SUV", "028": "SUV",
    "029": "Coupe", "030": "Sedan", "031": "Sedan", "032": "Sedan",
    "033": "Van / Minivan", "034": "Sedan", "035": "Sedan", "036": "Sedan",
    "037": "Sedan", "038": "SUV", "039": "Hatchback", "040": "SUV",
    "041": "Coupe", "042": "Sedan", "043": "Pickup Truck", "044": "Hatchback",
    "045": "SUV", "046": "Sedan", "047": "SUV", "048": "Coupe", "510": "Sedan"
}

FOLDER_PRIMARY_DAMAGE = {
    "001": "Scratch", "002": "No Damage", "003": "Scratch", "004": "Bumper Damage",
    "005": "Bumper Damage", "006": "Scratch", "007": "Dent", "008": "Major Smash / Collision",
    "009": "Bumper Damage", "010": "Scratch", "011": "Scratch", "012": "Dent",
    "013": "Bumper Damage", "014": "Scratch", "015": "Major Smash / Collision", "016": "Scratch",
    "017": "Dent", "018": "Dent", "019": "Bumper Damage", "020": "Bumper Damage",
    "021": "Scratch", "022": "Dent", "023": "Headlight / Taillight Damage", "024": "Scratch",
    "025": "Headlight / Taillight Damage", "026": "Bumper Damage", "027": "Dent", "028": "Scratch",
    "029": "Scratch", "030": "Bumper Damage", "031": "Headlight / Taillight Damage", "032": "Bumper Damage",
    "033": "Glass Crack / Shatter", "034": "Bumper Damage", "035": "Dent", "036": "Scratch",
    "037": "Bumper Damage", "038": "Dent", "039": "Major Smash / Collision", "040": "Major Smash / Collision",
    "041": "Headlight / Taillight Damage", "042": "Scratch", "043": "Dent", "044": "Scratch",
    "045": "Headlight / Taillight Damage", "046": "Dent", "047": "Scratch", "048": "Glass Crack / Shatter",
    "510": "Headlight / Taillight Damage"
}


def get_transforms(image_size=None, is_train=True):
    if image_size is None:
        image_size = config.IMAGE_SIZE
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def analyze_image_damage_features(full_path: str, folder_id: str, file_idx: int):
    primary_dmg = FOLDER_PRIMARY_DAMAGE.get(folder_id, "Scratch")
    try:
        img = cv2.imread(full_path)
        if img is None:
            return primary_dmg, 20.0
        small = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        std_intensity = float(np.std(gray)) / 255.0
        dark_ratio = float(np.count_nonzero(gray < 45)) / float(gray.size)
        bright_ratio = float(np.count_nonzero(gray > 215)) / float(gray.size)
    except Exception:
        edge_density = 0.08
        std_intensity = 0.2
        dark_ratio = 0.1
        bright_ratio = 0.05

    if file_idx == 0 and primary_dmg != "Major Smash / Collision" and edge_density < 0.07:
        return "No Damage", 0.0

    if primary_dmg == "No Damage":
        if edge_density > 0.14:
            return "Scratch", round(min(22.0, 8.0 + edge_density * 60.0), 1)
        return "No Damage", 0.0

    if primary_dmg == "Major Smash / Collision":
        pct = 48.0 + (edge_density * 180.0) + (dark_ratio * 35.0)
        return primary_dmg, round(min(88.0, max(42.0, pct)), 1)

    if primary_dmg == "Bumper Damage":
        pct = 18.0 + (edge_density * 130.0) + (std_intensity * 25.0)
        return primary_dmg, round(min(48.0, max(14.0, pct)), 1)

    if primary_dmg == "Headlight / Taillight Damage":
        pct = 15.0 + (bright_ratio * 90.0) + (edge_density * 110.0)
        return primary_dmg, round(min(42.0, max(12.0, pct)), 1)

    if primary_dmg == "Glass Crack / Shatter":
        pct = 22.0 + (edge_density * 140.0) + (bright_ratio * 60.0)
        return primary_dmg, round(min(52.0, max(18.0, pct)), 1)

    if primary_dmg == "Dent":
        pct = 14.0 + (dark_ratio * 45.0) + (edge_density * 95.0)
        return primary_dmg, round(min(38.0, max(10.0, pct)), 1)

    pct = 6.0 + (edge_density * 90.0) + (bright_ratio * 40.0)
    return "Scratch", round(min(26.0, max(5.0, pct)), 1)


class CarDamageDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame = None,
                 csv_path: str = None,
                 car_to_idx: dict = None,
                 damage_to_idx: dict = None,
                 transform=None,
                 cache_images=True):
        if df is not None:
            self.data = df.copy().reset_index(drop=True)
        elif csv_path and os.path.exists(csv_path):
            self.data = pd.read_csv(csv_path).reset_index(drop=True)
        else:
            raise ValueError("Either 'df' or a valid 'csv_path' must be provided.")

        self.transform = transform or get_transforms(is_train=False)
        self.cache_images = cache_images
        self.cached_pil = {}

        if car_to_idx is None:
            unique_cars = sorted(self.data['car_class'].astype(str).unique())
            self.car_to_idx = {c: i for i, c in enumerate(unique_cars)}
        else:
            self.car_to_idx = car_to_idx

        if damage_to_idx is None:
            unique_damages = sorted(self.data['damage_class'].astype(str).unique())
            self.damage_to_idx = {d: i for i, d in enumerate(unique_damages)}
        else:
            self.damage_to_idx = damage_to_idx

        self.idx_to_car = {v: k for k, v in self.car_to_idx.items()}
        self.idx_to_damage = {v: k for k, v in self.damage_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['image_path'])

        if self.cache_images and idx in self.cached_pil:
            image = self.cached_pil[idx]
        else:
            if not os.path.exists(img_path):
                image = Image.new("RGB", config.IMAGE_SIZE, color=(128, 128, 128))
            else:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = image.resize(config.IMAGE_SIZE)
                except Exception:
                    image = Image.new("RGB", config.IMAGE_SIZE, color=(128, 128, 128))
            if self.cache_images:
                self.cached_pil[idx] = image

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        car_name = str(row['car_class'])
        damage_name = str(row['damage_class'])

        car_label = self.car_to_idx.get(car_name, 0)
        damage_label = self.damage_to_idx.get(damage_name, 0)

        if 'damage_percent' in row and pd.notna(row['damage_percent']):
            try:
                damage_pct = float(row['damage_percent'])
            except ValueError:
                damage_pct = 0.0
        else:
            if damage_name.lower() in ["no damage", "clean", "none"]:
                damage_pct = 0.0
            elif "scratch" in damage_name.lower():
                damage_pct = 12.0
            elif "dent" in damage_name.lower():
                damage_pct = 25.0
            elif "crack" in damage_name.lower() or "glass" in damage_name.lower():
                damage_pct = 35.0
            elif "major" in damage_name.lower() or "smash" in damage_name.lower():
                damage_pct = 75.0
            else:
                damage_pct = 20.0

        percent_tensor = torch.tensor([damage_pct], dtype=torch.float32)

        return {
            "image": image_tensor,
            "car_label": torch.tensor(car_label, dtype=torch.long),
            "damage_label": torch.tensor(damage_label, dtype=torch.long),
            "damage_percent": percent_tensor,
            "image_path": img_path
        }


def scan_directory_to_dataframe(dataset_dir: str):
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    records = []

    for root, dirs, files in sorted(os.walk(dataset_dir)):
        img_files = sorted([
            f for f in files
            if any(f.lower().endswith(ext.replace('*', '')) for ext in valid_exts)
        ])
        for file_idx, file in enumerate(img_files):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, dataset_dir)
            parts = rel_path.split(os.sep)

            if len(parts) >= 3:
                car_class = parts[0]
                damage_class = parts[1]
                damage_pct = None
            elif len(parts) == 2:
                folder_name = parts[0]
                if folder_name.isdigit() or folder_name in FOLDER_CAR_MAP:
                    car_class = FOLDER_CAR_MAP.get(folder_name, "Sedan")
                    damage_class, damage_pct = analyze_image_damage_features(full_path, folder_name, file_idx)
                else:
                    car_class = "Sedan"
                    damage_class = folder_name
                    damage_pct = None
            else:
                car_class = "Sedan"
                damage_class = "Scratch"
                damage_pct = 12.0

            records.append({
                "image_path": full_path,
                "car_class": car_class,
                "damage_class": damage_class,
                "damage_percent": damage_pct
            })

    return pd.DataFrame(records)
