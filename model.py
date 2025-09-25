import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class RockfallModel:
    def __init__(self, path):
        self.model = load_model(path)
        self.labels = ["Stable", "Rockfall Risk"]  # adjust as per training

    def preprocess(self, image, target_size=(224,224)):
        image = image.convert("RGB").resize(target_size)
        arr = np.array(image).astype("float32")
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def predict(self, image):
        x = self.preprocess(image)
        preds = self.model.predict(x)[0]
        if len(preds) == 1:  # binary sigmoid
            prob = float(preds[0])
            label = self.labels[1] if prob > 0.5 else self.labels[0]
            return {"label": label, "probability": prob}
        else:  # multi-class softmax
            idx = int(np.argmax(preds))
            return {"label": self.labels[idx], "probability": float(preds[idx])}
