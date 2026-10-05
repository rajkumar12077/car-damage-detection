# AutoInspect AI: Car Damage Classification & % Calculation System

A complete Deep Learning system built **strictly from scratch** in PyTorch (without YOLO, ResNet, VGG, or any pretrained weights). It performs simultaneous multi-task classification and calculates vehicle damage percentage with visual spatial localization.

---

## Key Features

1. **Custom Multi-Task CNN Architecture (From Scratch)**:
   - 5 Convolutional feature extraction blocks with Batch Normalization, LeakyReLU, and Dropout.
   - **Task Head 1 (Car Classification)**: Multiclass classification for vehicle types (e.g. Sedan, SUV, Truck, Hatchback, etc.).
   - **Task Head 2 (Damage Classification)**: Multiclass classification for damage types (e.g. Dent, Scratch, Glass Crack, Bumper Damage, Major Collision, No Damage).
   - **Task Head 3 (Damage % Regression)**: Continuous severity prediction ($0.0\% - 100.0\%$).
   - **Zero Pretrained / Zero YOLO**: Built 100% with standard PyTorch layers.

2. **Damage Percentage Calculation Engine**:
   - **Grad-CAM Localization**: Automatically extracts activation maps from the final convolutional layer to identify the exact damaged vehicle surface.
   - **Surface Area Calculation**: Quantifies the proportion of affected surface area against vehicle dimensions.
   - **Blended Score**: Integrates spatial area percentage with model regression severity for high precision.
   - **Severity Tiers**: Automatically categorizes damage into *None (0%)*, *Superficial / Minor (1-20%)*, *Moderate (20-50%)*, or *Severe / Critical (>50%)*.

3. **Interactive Web Dashboard**:
   - Modern dark-mode UI powered by Flask, Tailwind CSS, and vanilla JS.
   - Drag-and-drop car photo upload or instant synthetic test generator.
   - Interactive SVG circular gauge for Damage %.
   - Multi-view toggler:
     - **Grad-CAM Heatmap Overlay**
     - **Original Vehicle Image**
     - **Isolated Damage Region Mask**
   - Probability distribution charts for both car types and damage categories.
   - Automated Body Shop repair advice & labor estimate.

---

## Project Structure

```text
aipro/
├── model.py              # CNN architecture built from scratch (CarDamageCNN)
├── config.py             # Hyperparameters, paths, and default class lists
├── dataset.py            # PyTorch Dataset loader (CSV & folder support)
├── train.py              # Multi-task training pipeline & checkpoint saver
├── generate_csv.py       # Helper utility to auto-scan image folders into CSV
├── app.py                # Flask Web Dashboard server & inference API
├── templates/
│   └── index.html        # Modern dashboard frontend
├── utils/
│   ├── __init__.py
│   └── cam.py            # Grad-CAM heatmap generator & damage % calculator
├── dataset/
│   └── README.md         # Instructions on organizing your custom dataset
├── models/
│   ├── best_car_damage_cnn.pth # Saved model weights
│   └── classes.json            # Dynamic class configuration
└── README.md             # Project documentation
```

---

## Quickstart: Running the Web Dashboard

To launch the web dashboard right now:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

You can upload any car image or click **"Scratch Test"**, **"Dent Test"**, or **"Collision Test"** on the dashboard to test the classification and percentage calculations immediately!

---

## How to Train on Your Own Dataset

You can supply your dataset in either of two ways:

### Option A: CSV Annotations (Recommended)
Create an `annotations.csv` file:

```csv
image_path,car_class,damage_class,damage_percent
data/car1.jpg,Sedan,Scratch,14.5
data/car2.jpg,SUV,Major Smash / Collision,72.0
data/car3.jpg,Truck,Dent,25.0
data/car4.jpg,Hatchback,No Damage,0.0
```
*(Note: `damage_percent` is optional. If omitted, the model uses damage category benchmarks and Grad-CAM area calculation).*

Train the model:
```bash
python train.py --data-csv path/to/annotations.csv --epochs 25 --batch-size 16
```

### Option B: Folder Structure
Organize your images into folders:

```text
my_dataset/
  Sedan/
    Scratch/
      car01.jpg
    Dent/
      car02.jpg
  SUV/
    Dent/
      car03.jpg
    No Damage/
      car04.jpg
```

Train directly on the folder:
```bash
python train.py --data-dir path/to/my_dataset --epochs 25
```

Alternatively, use `generate_csv.py` to auto-index your folders into a CSV first:
```bash
python generate_csv.py --input-dir path/to/my_dataset --output-csv dataset/annotations.csv
```

---

## Model Outputs & Inference API

When an image is submitted via the web dashboard or POST request to `/predict`, the system returns:

```json
{
  "success": true,
  "car_prediction": {
    "class": "Sedan",
    "confidence": 88.5,
    "breakdown": [...]
  },
  "damage_prediction": {
    "class": "Dent",
    "confidence": 92.1,
    "breakdown": [...]
  },
  "damage_calculation": {
    "final_percent": 24.3,
    "area_percent": 23.8,
    "regression_percent": 25.0,
    "severity": "Moderate"
  },
  "recommendation": {
    "action": "Paintless Dent Repair (PDR) / Panel Beating",
    "details": "Localized indentation impacting 24.3% area...",
    "estimated_labor_hours": "3 - 8 hours",
    "priority": "Medium"
  }
}
```
