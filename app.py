from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model("ovarian_cyst_detector.keras")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("image")

        for file in files:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            img = preprocess_image(path)
            pred = model.predict(img)[0][0]

            label = "🩸 Ovarian Cyst Detected" if pred < 0.5 else "✅ Normal Ovary"
            confidence = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)

            results.append({
                "image": path,
                "label": label,
                "confidence": confidence
            })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
