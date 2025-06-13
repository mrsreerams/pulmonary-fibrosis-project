# Pulmonary Fibrosis Detection Using Deep Learning 🫁

This project aims to detect pulmonary fibrosis from CT scan images using deep learning and machine learning models. It provides a user-friendly web interface for healthcare professionals to upload images and receive detection results with probability scores.

## 🔍 Features

- CT scan image classification for pulmonary fibrosis
- Implemented with three models:
  - InceptionV3 + Random Forest
  - VGG19 + SVM (**best performing**)
  - MobileNet + Random Forest
- Integrated Flask web app for easy image upload and prediction
- Clean and responsive user interface
- Displays model confidence level for each prediction

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn (SVM, Random Forest)
- Flask (Web Framework)
- OpenCV (Image Preprocessing)
- Streamlit (optional frontend prototype)

## 🗂️ Project Structure

.
├── models/ # Pre-trained models
├── app.py # Flask app file
├── templates/ # HTML templates
├── static/ # CSS/JS assets
├── utils.py # Image preprocessing & helpers
└── README.md


## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pulmonary-fibrosis-detector.git
   cd pulmonary-fibrosis-detector
Install dependencies
    pip install -r requirements.txt

Run the application
    python app.py

Open in browser
    http://localhost:5000

📊 Best Performing Model
VGG19 + SVM delivered the highest accuracy and reliability for detection.

🧑‍💻 Author
Venkata Sreeram Pachipulusu
