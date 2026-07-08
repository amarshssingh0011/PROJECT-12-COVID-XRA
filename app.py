from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the Keras model
model = tf.keras.models.load_model("my_model.keras")

@app.route("/")
def home():
    return "Model is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Example input:
        # {"features":[0.1,0.2,0.3,...]}
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
