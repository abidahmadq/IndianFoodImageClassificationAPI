from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'projectmodel.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Food classes
FOOD_CLASSES = ['Poha', 'SabudanaKhichdi', 'Samosa']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        return img_tensor
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

class ModelService:
    def __init__(self, model_path):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        with self.sess.as_default():
            with self.graph.as_default():
                self.model = load_model(model_path)
        logger.info("Model loaded successfully")

    def predict(self, image_tensor):
        """Make prediction using the loaded model"""
        with self.sess.as_default():
            with self.graph.as_default():
                predictions = self.model.predict(image_tensor)
                predicted_class = np.argmax(predictions[0])
                return predicted_class, predictions[0]

# Initialize model service
model_service = ModelService(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'classes': FOOD_CLASSES,
        'input_shape': (224, 224, 3),
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction endpoint"""
    if 'file' not in request.files:
        abort(400, description="No file provided")
    
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No file selected")

    if not allowed_file(file.filename):
        abort(400, description="File type not allowed")

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image and predict
        image_tensor = load_image(filepath)
        predicted_class, probabilities = model_service.predict(image_tensor)

        # Create response
        response = {
            "predicted_class": int(predicted_class),
            "predicted_food": FOOD_CLASSES[predicted_class],
            "probabilities": {
                food_class: float(prob)
                for food_class, prob in zip(FOOD_CLASSES, probabilities)
            }
        }

        # Cleanup
        os.remove(filepath)
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        abort(500, description="Error processing image")

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    if 'files[]' not in request.files:
        abort(400, description="No files provided")
    
    files = request.files.getlist('files[]')
    if not files:
        abort(400, description="No files selected")

    results = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_tensor = load_image(filepath)
            predicted_class, probabilities = model_service.predict(image_tensor)

            results.append({
                "filename": file.filename,
                "predicted_class": int(predicted_class),
                "predicted_food": FOOD_CLASSES[predicted_class],
                "probabilities": {
                    food_class: float(prob)
                    for food_class, prob in zip(FOOD_CLASSES, probabilities)
                }
            })

            os.remove(filepath)

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": "Failed to process image"
            })

    return jsonify({"results": results}), 200

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description)
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(error.description)
    }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)