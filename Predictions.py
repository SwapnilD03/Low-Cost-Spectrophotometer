from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model('path_to_your_model.h5')

# Load an image file
image_path = "C:\Users\SWAPNIL\OneDrive\Desktop\Transformer Oil Ageing\Training\Oil Ageing\Lightly Aged\Image48.jpg"
image = Image.open(image_path)

# Resize the image to match the input size of the model (e.g., 224x224 for many models)
image = image.resize((135, 85))

# Convert the image to array and normalize it
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0  # Normalize the image to the range [0, 1]

# Make predictions
predictions = model.predict(image)

# Assuming a single image, extract the predictions
predicted_class = np.argmax(predictions[0])

# Define the class names (this should match the classes used during training)
class_names = ['Fresh', 'Highly Aged', 'Lightly Aged', 'Moderately Aged']

# Get the predicted class name
predicted_class_name = class_names[predicted_class]

print(f"Predicted class: {predicted_class_name}")
