"""
classify.py
===========
A standalone utility to classify a single handwritten Arabic letter image
using our trained from-scratch NumPy Neural Network.

Usage:
    python classify.py path/to/your_image.png

To generate the weights file ('best_model_weights.npz') first, you can save 
the parameters of your best-fit model at the end of training by adding:
    np.savez('best_model_weights.npz', **gf_params)  # inside run_all.py
"""

import sys
import os
import numpy as np
from PIL import Image

# 1. Unicode & Transliterated class mapping for Task 7
CLASS_MAP = {
    0: ('ا', 'Alaf'),
    1: ('ب', 'Ba'),
    2: ('ت', 'Taa'),
    3: ('ث', 'Tha'),
    4: ('ج', 'Jeem'),
    5: ('ح', 'Haa'),
    6: ('خ', 'Kha'),
    7: ('د', 'Dal'),
    8: ('ذ', 'Thal'),
    9: ('ر', 'Raa')
}

# 2. Neural Network activation functions
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    """Simple forward pass through the 2-layer network."""
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return A2

def preprocess_image(img_path):
    """
    Preprocesses the custom input image exactly like our dataset loader:
    1. Opens the image and converts it to grayscale ('L' mode).
    2. Resizes it to 28x28 pixels.
    3. Flattens the image into a 1D vector of shape (784,).
    4. Normalizes pixel values to the range [0.0, 1.0].
    """
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        sys.exit(1)

    try:
        # Load and convert to grayscale
        img = Image.open(img_path).convert('L')
        # Resize to 28x28
        img = img.resize((28, 28))
        # Flatten and normalize
        arr = np.array(img, dtype=np.float64).flatten() / 255.0
        # Reshape to a batch of size 1 => (1, 784)
        return arr.reshape(1, -1)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if len(sys.argv) < 2:
        print("Usage: python classify.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    weights_path = 'best_model_weights.npz'

    # Check if trained weights are available
    if not os.path.exists(weights_path):
        print("=" * 60)
        print("WEIGHTS FILE NOT FOUND!")
        print("=" * 60)
        print(f"To classify images, you must first save your trained weights to '{weights_path}'.")
        print("\nYou can do this easily by adding this line at the end of run_all.py:")
        print("    np.savez('best_model_weights.npz', **gf_params)")
        print("\nOnce you run your training again, this file will be created and ready to use!")
        print("=" * 60)
        sys.exit(1)

    # Load weights
    print(f"Loading trained weights from '{weights_path}'...")
    data = np.load(weights_path)
    W1, b1 = data['W1'], data['b1']
    W2, b2 = data['W2'], data['b2']

    # Preprocess custom image
    print(f"Preprocessing custom image '{img_path}'...")
    X = preprocess_image(img_path)

    # Forward pass predictions
    A2 = forward(X, W1, b1, W2, b2)
    probabilities = A2[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100

    # Print results
    char, name = CLASS_MAP[predicted_class]
    print("\n" + "=" * 50)
    print(f"CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Predicted Letter : {char} ({name})")
    print(f"  Confidence Score : {confidence:.2f}%")
    print("=" * 50)

    # Print full confidence distribution
    print("\nFull Class Confidence Probabilities:")
    print("-" * 50)
    for idx, prob in enumerate(probabilities):
        c_char, c_name = CLASS_MAP[idx]
        bar = "█" * int(prob * 20)
        print(f"  Class {idx} {c_char} ({c_name:<5}): {prob*100:6.2f}% | {bar}")
    print("-" * 50)

if __name__ == "__main__":
    main()
