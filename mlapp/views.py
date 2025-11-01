import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid
import json

# Resolve base directory and preferred model/metrics paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Candidate model files in order of preference
_model_candidates = [
    os.path.join(BASE_DIR, 'models', 'mnist_model.h5'),
    os.path.join(BASE_DIR, 'models', 'mnist_best.keras'),
    os.path.join(BASE_DIR, 'mnist_model.h5'),
    os.path.join(BASE_DIR, 'mnist_best.keras'),
]

MODEL_PATH = next((p for p in _model_candidates if os.path.exists(p)), None)

# Load the trained MNIST model (accepts input (28,28,1)) if available
model = load_model(MODEL_PATH) if MODEL_PATH else None

# Try to read persisted metrics if available (from the improved trainer)
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')
if not os.path.exists(METRICS_PATH):
    METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')
_persisted_metrics = None
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH) as f:
            _persisted_metrics = json.load(f)
    except Exception:
        _persisted_metrics = None

# Optionally evaluate accuracy only if no persisted metrics are available
test_acc = None
try:
    if model is not None:
        # Defer evaluation until we know whether persisted metrics exist (see below)
        pass
except Exception:
    model = None

def preprocess_image(image_path):
    """
    Preprocess the uploaded image:
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values to [0, 1]
    - Invert colors if necessary
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = 1 - img_array  # Invert if necessary (black digits on white background)
    img_array = img_array.reshape(28, 28, 1)  # Reshape to (28, 28, 1) for the model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 28, 28, 1)
    return img_array

def home(request):
    predicted_class = None
    file_url = None
    accuracy = _persisted_metrics.get('test_accuracy') if _persisted_metrics else None
    message = "For best predictions, please upload clear, high-quality images of handwritten digits."

    # If we don't have persisted metrics but we do have a model, evaluate once to get accuracy
    global test_acc
    if accuracy is None and model is not None and test_acc is None:
        # Load and preprocess the MNIST dataset for accuracy calculation
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = (x_test / 255.0).astype('float32')
        x_test = x_test[..., None]
        # Evaluate the model on the test set to get accuracy
        try:
            _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        except Exception:
            test_acc = None
        accuracy = test_acc

    if model is None:
        missing = " or ".join([
            os.path.relpath(p, BASE_DIR) for p in _model_candidates
        ])
        message = (
            f"Model file not found. Place one of these files in your project: {missing}. "
            "You can train a new model by running mnist_model.py."
        )

    model_explanation = {
        'description': """
        This model is a compact convolutional neural network (CNN) trained on the MNIST dataset.
        It uses Conv-BatchNorm-ReLU blocks with MaxPooling and Dropout for regularization and is optimized
        to achieve high accuracy (>99%) while remaining fast for web inference.
        """,
        'technologies': """
        - TensorFlow/Keras for model training and inference.
        - NumPy for array manipulation and image preprocessing.
        - PIL (Python Imaging Library) for image processing.
        - Django for the web application.
        """,
        'code_snippets': [
            '''
            # Define the model architecture
            model = models.Sequential([
                layers.Flatten(input_shape=(28, 28)),  # Flatten input images to 1D
                layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
                layers.Dropout(0.2),  # Dropout for regularization
                layers.Dense(10)  # Output layer with 10 neurons (one for each digit)
            ])
            ''',
            '''
            # Compile and train the model
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5)  # Train the model for 5 epochs
            ''',
            '''
            # Preprocess the uploaded image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = img_array.reshape(28, 28, 1)  # Reshape for model input
            ''',
        ]
    }

    if request.method == 'POST' and request.FILES.get('image') and model is not None:
        uploaded_file = request.FILES['image']
        
        # Generate a safe, unique filename
        safe_filename = f"{uuid.uuid4()}{os.path.splitext(uploaded_file.name)[1]}"
        file_path = os.path.join(settings.MEDIA_ROOT, safe_filename)

        # Save the uploaded file securely
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Predict the digit
        predictions = model.predict(processed_image)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions) * 100.0)

        # Prepare the file URL for displaying the image
        file_url = settings.MEDIA_URL + safe_filename

    # If user posted an image but model is missing, surface a friendly error
    error_message = None
    if request.method == 'POST' and request.FILES.get('image') and model is None:
        error_message = "Model file is missing, so we couldn't run a prediction. Please add a model file and reload."

    return render(request, 'home.html', {
        'file_url': file_url,
        'predicted_class': predicted_class,
        'confidence': confidence if 'confidence' in locals() else None,
        'accuracy': (accuracy * 100.0) if isinstance(accuracy, (int, float)) else accuracy,
        'message': message,
        'error_message': error_message,
        'model_explanation': model_explanation  # Pass the explanation to the template
    })