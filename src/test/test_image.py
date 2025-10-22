from src.predictor.image_predictor import ImagePredictor


IMAGE_PATH="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\processed_binary\\normal\\Parabasal_001_01.bmp"

predictor = ImagePredictor()

label, prob = predictor.predict(IMAGE_PATH)
print(f" Prediction: {label}")
print(f"Confidence → Abnormal: {prob[0]*100:.2f}% | Normal: {prob[1]*100:.2f}%")