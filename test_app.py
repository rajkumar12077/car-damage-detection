import io
import json
from PIL import Image, ImageDraw
import torch

from app import app, load_cnn_model, model, cam_analyzer
import config

def run_tests():
    print("[+] Testing Car Damage CNN...")
    assert model is not None, "Model failed to load"
    print(f"[OK] Model loaded on {config.DEVICE}")

    test_img = Image.new("RGB", (224, 224), color=(60, 80, 110))
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([40, 100, 180, 160], fill=(200, 50, 50))
    
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with app.test_client() as client:
        res = client.get("/")
        assert res.status_code == 200, f"Index failed: {res.status_code}"
        print("[OK] Index page '/' returns 200 OK")

        res_sample = client.get("/api/generate_sample?type=dent")
        assert res_sample.status_code == 200, "Sample generator failed"
        data_sample = json.loads(res_sample.data)
        assert "image_data" in data_sample
        print("[OK] '/api/generate_sample' returns synthetic car")

        data = {
            'image': (io.BytesIO(img_bytes.getvalue()), 'test.jpg'),
            'threshold': '0.35'
        }
        res_predict = client.post("/predict", data=data, content_type='multipart/form-data')
        assert res_predict.status_code == 200, f"Predict failed: {res_predict.status_code}, {res_predict.data}"
        pred_json = json.loads(res_predict.data)
        
        assert pred_json["success"] is True
        print(f"[OK] Car Class: {pred_json['car_prediction']['class']} ({pred_json['car_prediction']['confidence']}%)")
        print(f"[OK] Damage Class: {pred_json['damage_prediction']['class']} ({pred_json['damage_prediction']['confidence']}%)")
        print(f"[OK] Calculated Damage %: {pred_json['damage_calculation']['final_percent']}% (Severity: {pred_json['damage_calculation']['severity']})")
        print(f"[OK] Grad-CAM overlay image generated (base64 length: {len(pred_json['images']['heatmap'])})")
        print(f"[OK] Repair Recommendation: {pred_json['recommendation']['action']}")

    print("\n[OK] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
