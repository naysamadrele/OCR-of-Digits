<h1 align="center">🔢 Digit Recognizer with OpenCV + KNN</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv">
  <img src="https://img.shields.io/badge/KNN-Classifier-yellow?logo=scikit-learn">
</p>

<p align="center">
  🎯 A fun and simple way to recognize handwritten digits using image processing and machine learning.
</p>

---

## 🌟 Overview

This project uses the [scikit-learn `digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset to train a K-Nearest Neighbors (KNN) classifier, then uses **OpenCV** to detect and recognize digits from custom input images.

---

## 📸 Demo Output

<img src="sample_output.png" width="600"/>

*Sample: Detected digits are boxed and labeled automatically!*

---

## 📂 Features

✅ Real-time digit recognition from images  
✅ Preprocessing with OpenCV (grayscale, thresholding, contour detection)  
✅ KNN model trained on built-in digits dataset  
✅ Visual annotations of predictions  
✅ Lightweight and beginner-friendly code  

---

## 🧰 Tech Stack

- Python 3.8+
- OpenCV
- scikit-learn
- NumPy

---

## 🧑‍💻 How to Run on Any Desktop

### 🚀 Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/digit-recognition-knn.git
cd digit-recognition-knn
🛠️ Step 2: Install Requirements
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📷 Step 3: Add Your Own Image
Place your image (containing handwritten digits) in the project folder. Update the filename in the main() call inside digit_ocr.py:

python
Copy
Edit
main(r'your_image.jpg')
▶️ Step 4: Run the Script
bash
Copy
Edit
python digit_ocr.py
A window will pop up showing the image with recognized digits.

📁 File Structure
bash
Copy
Edit
digit-recognition-knn/
├── digit_ocr.py           # Main Python script
├── sample_output.png      # Demo image
├── requirements.txt       # Python dependencies
└── README.md              # This file
📦 requirements.txt
If you don't have it, here’s what to put:

Copy
Edit
opencv-python
scikit-learn
numpy
Save this as requirements.txt.

📌 Notes
Best results are achieved with black digits on a white background.

Works with printed or handwritten digits, provided they are spaced apart.

📄 License
This project is open-source and licensed under the MIT License.

❤️ Credits
Built with 🧠 scikit-learn and 👁️ OpenCV.
Inspired by classic digit classification use cases.

✨ Like the project?
Give it a ⭐️ on GitHub and share it with your friends!
