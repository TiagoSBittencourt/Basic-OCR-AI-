# 🧠 OCR_AI – Neural Network for Handwritten Digit Recognition

OCR_AI is a simple Python-based neural network designed to recognize handwritten digits drawn on a 20x20 canvas. The network uses NumPy and is fully self-contained—no deep learning frameworks like TensorFlow or PyTorch required. It’s ideal for educational purposes or as a base for more complex OCR systems.

---

## 📸 Example Use Case

The model takes a 20x20 grayscale image (flattened into a 400-dimensional vector), processes it through a single hidden layer, and outputs the predicted digit (0–9).

---

## 📁 Project Structure

OCR_AI/  
│  
├── src/  
│ ├── ocr.py # Neural network logic  
│ ├── server.py # Web basic server to interact with the model  
│ ├── ocr.js # Input preprocessing   
│ ├── ocr.html # Page Structure  
│ └── neural_network.json # Stored model weights   
│  
└── README.md  
└── .gitignore  


---

## 🔧 Features

- Simple feedforward neural network with:
  - Input layer: 400 neurons (20x20 pixels)
  - Hidden layer: customizable size (default: 25 neurons)
  - Output layer: 10 neurons (digits 0 to 9)
- Supports training on a single image at a time (`train_on_instance`)
- JSON-based persistence (`neural_network.json`)
- Activation functions: `sigmoid` by default (ReLU supported)
- Written using only standard Python + NumPy

---

## 🚀 Run the Server

- Simply run:

  - python src/server.py


