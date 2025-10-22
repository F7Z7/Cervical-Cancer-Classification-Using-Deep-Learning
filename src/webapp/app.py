
from flask import Flask, render_template, request, url_for
import os

from src.predictor.image_predictor import ImagePredictor

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

predictor = ImagePredictor()


@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("upload.html", error="Please select a file.")

        file=request.files["image"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)


        label,prob = predictor.predict(file_path)
        prob_text = f"Abnormal: {prob[0] * 100:.2f}% | Normal: {prob[1] * 100:.2f}%"

        image_url = url_for('static', filename=f'uploads/{file.filename}')

        return render_template("predictor.html",prediction=label,probability=prob_text)

    return render_template("predictor.html")


if __name__ == '__main__':
    app.run(debug=True)