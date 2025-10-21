import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch


MODEL_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\src\\classical_models\\models\\logistic_model.joblib"
SCALARS_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\src\\classical_models\\models\\scalar.joblib"
IMAGE_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\processed_binary\\abnormal\\Dyskeratotic_002_01.bmp"

logistic_model = joblib.load(MODEL_PATH)
scalar_model = joblib.load(SCALARS_PATH)


resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove last FC layer
resnet.eval()

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

resnet.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert('RGB')
img_t = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    feats = resnet(img_t)
    feats = feats.cpu().numpy().reshape(1, -1)


feats_scaled=scalar_model.transform(feats)
pred = logistic_model.predict(feats_scaled)[0]
prob = logistic_model.predict_proba(feats_scaled)[0]
label_map = {0: "Abnormal", 1: "Normal"}  # adjust if class mapping differs
print(f"c Prediction: {label_map[pred]}")
print(f"Confidence → Abnormal: {prob[0]:.3f}, Normal: {prob[1]:.3f}")