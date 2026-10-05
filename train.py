import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from model import build_model
from dataset import CarDamageDataset, scan_directory_to_dataframe, get_transforms


def train_one_epoch(model, dataloader, optimizer, criterion_car, criterion_damage, criterion_percent, device):
    model.train()
    running_loss = 0.0
    correct_car = 0
    correct_damage = 0
    total_samples = 0
    abs_percent_error = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        car_labels = batch["car_label"].to(device)
        damage_labels = batch["damage_label"].to(device)
        damage_percents = batch["damage_percent"].to(device)

        optimizer.zero_grad()

        car_logits, damage_logits, pred_percents = model(images)

        loss_car = criterion_car(car_logits, car_labels)
        loss_damage = criterion_damage(damage_logits, damage_labels)
        loss_percent = criterion_percent(pred_percents, damage_percents)

        total_loss = (config.LOSS_WEIGHT_CAR * loss_car +
                      config.LOSS_WEIGHT_DAMAGE * loss_damage +
                      config.LOSS_WEIGHT_PERCENT * loss_percent)

        total_loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size

        _, pred_car_cls = torch.max(car_logits, 1)
        _, pred_dmg_cls = torch.max(damage_logits, 1)

        correct_car += (pred_car_cls == car_labels).sum().item()
        correct_damage += (pred_dmg_cls == damage_labels).sum().item()
        abs_percent_error += torch.abs(pred_percents - damage_percents).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    car_acc = (correct_car / total_samples) * 100.0
    damage_acc = (correct_damage / total_samples) * 100.0
    mae_percent = abs_percent_error / total_samples

    return epoch_loss, car_acc, damage_acc, mae_percent


def validate(model, dataloader, criterion_car, criterion_damage, criterion_percent, device):
    model.eval()
    running_loss = 0.0
    correct_car = 0
    correct_damage = 0
    total_samples = 0
    abs_percent_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            car_labels = batch["car_label"].to(device)
            damage_labels = batch["damage_label"].to(device)
            damage_percents = batch["damage_percent"].to(device)

            car_logits, damage_logits, pred_percents = model(images)

            loss_car = criterion_car(car_logits, car_labels)
            loss_damage = criterion_damage(damage_logits, damage_labels)
            loss_percent = criterion_percent(pred_percents, damage_percents)

            total_loss = (config.LOSS_WEIGHT_CAR * loss_car +
                          config.LOSS_WEIGHT_DAMAGE * loss_damage +
                          config.LOSS_WEIGHT_PERCENT * loss_percent)

            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size

            _, pred_car_cls = torch.max(car_logits, 1)
            _, pred_dmg_cls = torch.max(damage_logits, 1)

            correct_car += (pred_car_cls == car_labels).sum().item()
            correct_damage += (pred_dmg_cls == damage_labels).sum().item()
            abs_percent_error += torch.abs(pred_percents - damage_percents).sum().item()
            total_samples += batch_size

    val_loss = running_loss / (total_samples if total_samples > 0 else 1)
    car_acc = (correct_car / (total_samples if total_samples > 0 else 1)) * 100.0
    damage_acc = (correct_damage / (total_samples if total_samples > 0 else 1)) * 100.0
    mae_percent = abs_percent_error / (total_samples if total_samples > 0 else 1)

    return val_loss, car_acc, damage_acc, mae_percent


def main():
    parser = argparse.ArgumentParser(description="Train Car Damage Classification CNN from Scratch")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to CSV with [image_path, car_class, damage_class, damage_percent]")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to directory containing images organized in folders")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default=config.MODEL_PATH, help="Path to save best weights")
    args = parser.parse_args()

    print("=" * 60)
    print("      Car Damage Classification & Percentage CNN")
    print("                 Training Pipeline")
    print("=" * 60)

    df = None
    if args.data_csv and os.path.exists(args.data_csv):
        print(f"[+] Loading dataset from CSV: {args.data_csv}")
        df = pd.read_csv(args.data_csv)
    elif args.data_dir and os.path.isdir(args.data_dir):
        print(f"[+] Scanning images from directory: {args.data_dir}")
        df = scan_directory_to_dataframe(args.data_dir)
        print(f"    Found {len(df)} images.")
    else:
        default_csv = os.path.join(config.BASE_DIR, "dataset", "annotations.csv")
        default_dir = os.path.join(config.BASE_DIR, "dataset", "images")
        if os.path.exists(default_csv):
            print(f"[+] Found default annotations CSV: {default_csv}")
            df = pd.read_csv(default_csv)
        elif os.path.exists(default_dir):
            print(f"[+] Found default images directory: {default_dir}")
            df = scan_directory_to_dataframe(default_dir)
        else:
            print("[!] No dataset specified or found.")
            print("\nHow to provide your dataset:")
            print("  Option 1: python train.py --data-csv path/to/annotations.csv")
            print("  Option 2: python train.py --data-dir path/to/dataset_folder")
            print("  Option 3: Place your images inside the 'dataset/' folder in this directory.")
            print("\nCSV format expected:")
            print("  image_path,car_class,damage_class,damage_percent")
            print("  C:/images/car1.jpg,Sedan,Dent,25.0")
            print("=" * 60)
            sys.exit(0)

    if df is None or len(df) == 0:
        print("[!] Dataset is empty. Please verify your file paths.")
        sys.exit(1)

    car_classes = sorted(df['car_class'].astype(str).unique().tolist())
    damage_classes = sorted(df['damage_class'].astype(str).unique().tolist())

    car_to_idx = {c: i for i, c in enumerate(car_classes)}
    damage_to_idx = {d: i for i, d in enumerate(damage_classes)}

    print(f"\n[+] Detected {len(car_classes)} Car Classes: {car_classes}")
    print(f"[+] Detected {len(damage_classes)} Damage Classes: {damage_classes}")
    print(f"[+] Total Samples: {len(df)}")

    os.makedirs(os.path.dirname(config.CLASSES_PATH), exist_ok=True)
    with open(config.CLASSES_PATH, "w") as f:
        json.dump({
            "car_classes": car_classes,
            "damage_classes": damage_classes
        }, f, indent=4)
    print(f"[+] Saved class configuration to {config.CLASSES_PATH}")

    full_dataset = CarDamageDataset(
        df=df,
        car_to_idx=car_to_idx,
        damage_to_idx=damage_to_idx,
        transform=get_transforms(config.IMAGE_SIZE, is_train=True)
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset, val_dataset = full_dataset, full_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[+] Train Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    device = torch.device(config.DEVICE)
    print(f"[+] Using Device: {device}")

    model = build_model(
        num_car_classes=len(car_classes),
        num_damage_classes=len(damage_classes),
        device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    criterion_car = nn.CrossEntropyLoss()
    criterion_damage = nn.CrossEntropyLoss()
    criterion_percent = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("\n[+] Starting Training Loop...")
    print("-" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Car Acc':<8} | {'Dmg Acc':<8} | {'Val Loss':<10} | {'Val Car%':<9} | {'Val Dmg%':<9} | {'Val MAE%':<8}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_car_acc, tr_dmg_acc, tr_mae = train_one_epoch(
            model, train_loader, optimizer,
            criterion_car, criterion_damage, criterion_percent, device
        )

        vl_loss, vl_car_acc, vl_dmg_acc, vl_mae = validate(
            model, val_loader,
            criterion_car, criterion_damage, criterion_percent, device
        )

        scheduler.step(vl_loss)

        print(f"{epoch:<6} | {tr_loss:<11.4f} | {tr_car_acc:<7.1f}% | {tr_dmg_acc:<7.1f}% | {vl_loss:<10.4f} | {vl_car_acc:<8.1f}% | {vl_dmg_acc:<8.1f}% | {vl_mae:<7.1f}%")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'car_classes': car_classes,
                'damage_classes': damage_classes,
                'best_val_loss': best_val_loss
            }, args.output)
            print(f"       -> Saved Best Model Checkpoint to {args.output}")

    print("=" * 80)
    print(f"[OK] Training Completed! Best model saved at: {args.output}")


if __name__ == "__main__":
    main()
