
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model("cnn_mnist_model.h5")

# Streamlit App
st.title("MNIST Digit Recognition with CNN")
st.write("Upload a grayscale image of a handwritten digit (28x28 pixels).")

# Upload Image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((28, 28))  # Resize to 28x28 pixels
    img_array = img_to_array(img).astype("float32") / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # Show results
    st.write(f"**Predicted Digit:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

# Evaluate the model accuracy
st.write("---")
if st.button("Show Model Accuracy"):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"**Model Test Accuracy:** {accuracy:.2f}")
