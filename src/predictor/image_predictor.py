import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image


class ImagePredictor:
    def __init__(self,
                 MODEL_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\src\\classical_models\\models\\logistic_model.joblib",
                 SCALARS_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\src\\classical_models\\models\\scalar.joblib"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logistic_model = joblib.load(MODEL_PATH)
        self.scalar_model = joblib.load(SCALARS_PATH)

        self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) #last layer removed
        self.resnet.eval().to(self.device)

        self.transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.label_map = {0: "Abnormal", 1: "Normal"}



    def feature_extract(self, img):
        img_t=self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.resnet(img_t)
        feats = feats.squeeze().cpu().numpy().reshape(1, -1)
        return feats
    def predict(self, image_path):
        img=Image.open(image_path).convert('RGB')
        features=self.feature_extract(img)
        features_scaled = self.scalar_model.transform(features)

        pred = self.logistic_model.predict(features_scaled)[0]
        prob = self.logistic_model.predict_proba(features_scaled)[0]
        label = self.label_map[pred]
        return label, prob