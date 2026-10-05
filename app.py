import os
import io
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

import config
from model import build_model
from utils.cam import GradCAMDamageAnalyzer
from dataset import get_transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

device = torch.device(config.DEVICE)
model = None
cam_analyzer = None
car_classes = config.DEFAULT_CAR_CLASSES
damage_classes = config.DEFAULT_DAMAGE_CLASSES
is_trained_checkpoint = False


def load_cnn_model():
    global model, cam_analyzer, car_classes, damage_classes, is_trained_checkpoint

    if os.path.exists(config.CLASSES_PATH):
        try:
            with open(config.CLASSES_PATH, "r") as f:
                classes_data = json.load(f)
                car_classes = classes_data.get("car_classes", config.DEFAULT_CAR_CLASSES)
                damage_classes = classes_data.get("damage_classes", config.DEFAULT_DAMAGE_CLASSES)
                print(f"[+] Loaded custom classes from {config.CLASSES_PATH}")
        except Exception as e:
            print(f"[!] Warning reading classes.json: {e}")

    model = build_model(
        num_car_classes=len(car_classes),
        num_damage_classes=len(damage_classes),
        device=device
    )

    if os.path.exists(config.MODEL_PATH):
        try:
            checkpoint = torch.load(config.MODEL_PATH, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            is_trained_checkpoint = True
            print(f"[OK] Loaded trained model weights from {config.MODEL_PATH}")
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}. Running with scratch weights.")
            is_trained_checkpoint = False
    else:
        print("[i] Notice: No trained checkpoint found at models/best_car_damage_cnn.pth.")
        print("[i] Model is running from scratch initialization. Train using 'python train.py'.")
        is_trained_checkpoint = False

    model.eval()
    cam_analyzer = GradCAMDamageAnalyzer(model, device=device)


load_cnn_model()


def pil_to_base64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mime_type = "image/png" if format == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{img_str}"


def generate_repair_recommendation(damage_class: str, damage_percent: float, severity: str):
    dmg_lower = damage_class.lower()

    if "no damage" in dmg_lower or severity == "None":
        return {
            "action": "Vehicle in Good Condition",
            "details": "No visible structural, cosmetic, or panel damage detected. No immediate bodywork or mechanical intervention required.",
            "estimated_labor_hours": "0 hours",
            "priority": "Low",
            "badge_color": "green"
        }
    elif "scratch" in dmg_lower:
        return {
            "action": "Detailing & Paint Touch-Up / Respray",
            "details": f"Cosmetic scratch affecting approximately {damage_percent}% surface area. Recommendation: Surface compound buffing, color-matched primer and clear-coat refinishing.",
            "estimated_labor_hours": "2 - 5 hours",
            "priority": "Medium",
            "badge_color": "yellow"
        }
    elif "dent" in dmg_lower:
        return {
            "action": "Paintless Dent Repair (PDR) / Panel Beating",
            "details": f"Localized indentation impacting {damage_percent}% area. If paint film is intact, PDR is recommended. Otherwise, standard panel reshaping and repaint.",
            "estimated_labor_hours": "3 - 8 hours",
            "priority": "Medium",
            "badge_color": "amber"
        }
    elif "bumper" in dmg_lower:
        return {
            "action": "Bumper Realignment / Plastic Weld or Replacement",
            "details": f"Bumper damage calculated at {damage_percent}%. Inspect internal absorbing foam and sensor mounts (parking sensors/radar) for hidden misalignment.",
            "estimated_labor_hours": "4 - 10 hours",
            "priority": "High" if damage_percent > 30 else "Medium",
            "badge_color": "orange"
        }
    elif "glass" in dmg_lower or "crack" in dmg_lower:
        return {
            "action": "Glass Replacement or Resin Sealant",
            "details": f"Glazing breach detected ({damage_percent}%). Small chips may be resin-repaired; larger stress fractures necessitate complete glass replacement with ADAS recalibration.",
            "estimated_labor_hours": "2 - 4 hours",
            "priority": "High",
            "badge_color": "red"
        }
    elif "major" in dmg_lower or "smash" in dmg_lower or damage_percent >= 50.0:
        return {
            "action": "Extensive Body Shop Frame & Panel Overhaul",
            "details": f"Severe structural/chassis damage detected ({damage_percent}% impact index). Requires full bench frame measuring, crumple-zone alignment, and multiple component swaps.",
            "estimated_labor_hours": "25 - 60+ hours",
            "priority": "Critical",
            "badge_color": "red"
        }
    else:
        return {
            "action": "General Body Shop Inspection",
            "details": f"Detected {damage_class} with {damage_percent}% estimated severity. Comprehensive physical inspection recommended.",
            "estimated_labor_hours": "4 - 12 hours",
            "priority": "Medium",
            "badge_color": "orange"
        }


@app.route("/")
def index():
    return render_template(
        "index.html",
        car_classes=car_classes,
        damage_classes=damage_classes,
        is_trained=is_trained_checkpoint,
        device_name=str(device).upper()
    )


@app.route("/api/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "car_classes": car_classes,
        "damage_classes": damage_classes,
        "is_trained": is_trained_checkpoint
    })


@app.route("/api/generate_sample", methods=["GET"])
def generate_sample_image():
    sample_type = request.args.get("type", "dent")
    img = Image.new("RGB", (400, 300), color=(240, 243, 246))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([40, 130, 360, 220], radius=15, fill=(70, 90, 120), outline=(40, 50, 70), width=3)
    draw.polygon([(90, 130), (140, 70), (270, 70), (320, 130)], fill=(85, 105, 140), outline=(40, 50, 70))
    draw.polygon([(105, 125), (145, 78), (200, 78), (200, 125)], fill=(180, 215, 245))
    draw.polygon([(210, 125), (210, 78), (265, 78), (305, 125)], fill=(180, 215, 245))
    draw.ellipse([80, 195, 140, 255], fill=(30, 30, 30), outline=(10, 10, 10), width=3)
    draw.ellipse([98, 213, 122, 237], fill=(180, 180, 180))
    draw.ellipse([260, 195, 320, 255], fill=(30, 30, 30), outline=(10, 10, 10), width=3)
    draw.ellipse([278, 213, 302, 237], fill=(180, 180, 180))

    if sample_type == "dent":
        draw.arc([160, 140, 230, 190], start=30, end=210, fill=(20, 20, 30), width=5)
        draw.ellipse([180, 155, 210, 175], fill=(45, 60, 85))
    elif sample_type == "scratch":
        draw.line([140, 145, 240, 160], fill=(220, 50, 50), width=3)
        draw.line([150, 155, 230, 175], fill=(220, 50, 50), width=2)
        draw.line([160, 140, 210, 150], fill=(220, 50, 50), width=2)
    elif sample_type == "major":
        draw.polygon([(320, 130), (370, 150), (350, 210), (310, 220)], fill=(180, 40, 40))
        draw.line([(310, 140), (370, 200)], fill=(20, 20, 20), width=4)
        draw.line([(340, 120), (330, 210)], fill=(20, 20, 20), width=4)

    return jsonify({
        "image_data": pil_to_base64(img, format="PNG")
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model, cam_analyzer

    if 'image' not in request.files and 'image_base64' not in request.form:
        return jsonify({"error": "No image uploaded. Please upload a vehicle image."}), 400

    threshold = float(request.form.get("threshold", 0.35))

    try:
        if 'image' in request.files:
            file = request.files['image']
            original_image = Image.open(file.stream).convert("RGB")
        else:
            base64_data = request.form['image_base64']
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            image_bytes = base64.b64decode(base64_data)
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    transform = get_transforms(config.IMAGE_SIZE, is_train=False)
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    car_logits, damage_logits, pred_percent_tensor = model(input_tensor)

    car_probs = F.softmax(car_logits, dim=1).detach().cpu().squeeze().numpy()
    damage_probs = F.softmax(damage_logits, dim=1).detach().cpu().squeeze().numpy()

    car_idx = int(np.argmax(car_probs))
    damage_idx = int(np.argmax(damage_probs))

    car_predicted = car_classes[car_idx] if car_idx < len(car_classes) else f"Car Class #{car_idx}"
    car_conf = float(car_probs[car_idx]) * 100.0

    damage_predicted = damage_classes[damage_idx] if damage_idx < len(damage_classes) else f"Damage Class #{damage_idx}"
    damage_conf = float(damage_probs[damage_idx]) * 100.0

    regressed_percent = float(pred_percent_tensor.item())

    heatmap = cam_analyzer.generate_heatmap(input_tensor, class_idx=damage_idx)

    damage_metrics = cam_analyzer.calculate_damage_percentage(
        heatmap=heatmap,
        damage_class_name=damage_predicted,
        regression_percent=regressed_percent,
        threshold=threshold
    )

    overlay_img = cam_analyzer.overlay_heatmap_on_image(original_image, heatmap, alpha=0.45)

    mask_resized = (np.array(Image.fromarray(damage_metrics["mask"]).resize(original_image.size, Image.NEAREST)) * 255).astype(np.uint8)
    mask_colored = Image.fromarray(mask_resized).convert("RGB")

    recommendation = generate_repair_recommendation(
        damage_class=damage_predicted,
        damage_percent=damage_metrics["final_percent"],
        severity=damage_metrics["severity"]
    )

    car_breakdown = [
        {"class": car_classes[i] if i < len(car_classes) else f"Class {i}", "prob": round(float(prob) * 100.0, 1)}
        for i, prob in enumerate(car_probs)
    ]
    car_breakdown.sort(key=lambda x: x["prob"], reverse=True)

    damage_breakdown = [
        {"class": damage_classes[i] if i < len(damage_classes) else f"Class {i}", "prob": round(float(prob) * 100.0, 1)}
        for i, prob in enumerate(damage_probs)
    ]
    damage_breakdown.sort(key=lambda x: x["prob"], reverse=True)

    return jsonify({
        "success": True,
        "is_trained": is_trained_checkpoint,
        "car_prediction": {
            "class": car_predicted,
            "confidence": round(car_conf, 1),
            "breakdown": car_breakdown
        },
        "damage_prediction": {
            "class": damage_predicted,
            "confidence": round(damage_conf, 1),
            "breakdown": damage_breakdown
        },
        "damage_calculation": {
            "final_percent": damage_metrics["final_percent"],
            "area_percent": damage_metrics["area_percent"],
            "regression_percent": damage_metrics["regression_percent"],
            "severity": damage_metrics["severity"]
        },
        "recommendation": recommendation,
        "images": {
            "original": pil_to_base64(original_image, format="JPEG"),
            "heatmap": pil_to_base64(overlay_img, format="JPEG"),
            "mask": pil_to_base64(mask_colored, format="PNG")
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Starting Car Damage AI Dashboard (Flask Web Server)")
    print("  Device:", device)
    print("  Access the dashboard at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
