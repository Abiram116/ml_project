import tensorflow as tf  # Add this import
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid

# Load the trained MNIST model
model = load_model('mnist_model.h5')

# Load and preprocess the MNIST dataset for accuracy calculation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Using tf here
x_test = x_test / 255.0  # Normalize test data

# Evaluate the model on the test set to get accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

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
    accuracy = test_acc  # Pass the accuracy to the template
    message = "For best predictions, please upload clear, high-quality images of handwritten digits."

    model_explanation = {
        'description': """
        This model is a simple neural network trained on the MNIST dataset, a large database of handwritten digits. 
        The model is a feed-forward neural network with one hidden layer and uses the ReLU activation function.
        The model was trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.
        """,
        'technologies': """
        - TensorFlow/Keras for building and training the model.
        - NumPy for array manipulation and image preprocessing.
        - PIL (Python Imaging Library) for image processing.
        - Django for creating the web application.
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

    if request.method == 'POST' and request.FILES.get('image'):
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
        predicted_class = np.argmax(predictions)

        # Prepare the file URL for displaying the image
        file_url = settings.MEDIA_URL + safe_filename

    return render(request, 'home.html', {
        'file_url': file_url,
        'predicted_class': predicted_class,
        'accuracy': accuracy,
        'message': message,
        'model_explanation': model_explanation  # Pass the explanation to the template
    })