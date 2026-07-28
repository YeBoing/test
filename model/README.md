---
license: mit
language:
- en
base_model:
- sakshamkr1/ResNet50-APTOS-DR
pipeline_tag: image-classification
tags:
- onnx
---
# ResNet50-APTOS-DR (ONNX) 
**5-class Diabetic Retinopathy classifier** ready for edge devices.

**Original model**: sakshamkr1/ResNet50-APTOS-DR  
**Format**: ONNX 
**Input shape**: (batch, 3, 224, 224) RGB fundus image  
**Output**: 5 classes (APTOS 2019)

### Classes
- 0: No DR  
- 1: Mild DR  
- 2: Moderate DR  
- 3: Severe DR  
- 4: Proliferative DR  

### Perfect 
- Model size: ~105 MB (single file)  
- RAM usage: ~150-220 MB  
- Speed: ~0.8–1.5 seconds per image on CPU

### Quick test code for colab

```python
# ============================
# 1. Install dependencies
# ============================
!pip install -q onnxruntime huggingface_hub pillow torchvision matplotlib

# ============================
# 2. Download the ONNX model
# ============================
from huggingface_hub import hf_hub_download

print("📥 Downloading iris-vit.onnx ...")
model_path = hf_hub_download(
    repo_id="Shadow0482/iris-onnx",
    filename="iris-vit.onnx"
)
print(f"✅ Model downloaded: {model_path}")

# ============================
# 3. Load model & define inference
# ============================
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from google.colab import files

# Load ONNX session (CPU is fine & fast for this ~105 MB model)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Preprocessing (exactly what the model expects)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("✅ Model loaded successfully!")

# ============================
# 4. Upload a fundus image & run inference
# ============================
print("\n📤 Please upload a color fundus/retina image (JPG/PNG)...")
uploaded = files.upload()

if uploaded:
    img_path = list(uploaded.keys())[0]
    img = Image.open(img_path).convert("RGB")
    
    # Preprocess
    input_tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
    
    # Inference
    outputs = session.run(None, {"input": input_tensor})[0][0]
    
    # Softmax
    exp_scores = np.exp(outputs)
    probs = exp_scores / np.sum(exp_scores)
    pred_idx = np.argmax(probs)
    
    classes = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    
    print(f"\n🎯 **Prediction:** {classes[pred_idx]}")
    print(f"   Confidence: {probs[pred_idx]*100:.1f}%")
    print("\n📊 Full probabilities:")
    for name, p in zip(classes, probs):
        print(f"   {name:20} → {p*100:5.1f}%")
    
    # Show image
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {classes[pred_idx]} ({probs[pred_idx]*100:.1f}%)", fontsize=14)
    plt.axis("off")
    plt.show()
```

**License**: MIT  
Made for low-resource diabetic retinopathy screening ❤️