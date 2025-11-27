# Web-based Face Emotion Detection System

### Project Description

In this project, students shall build a **web-based real-time facial emotion detection system** using:

- **Flask** (Python backend)  
- **HTML/CSS/JavaScript** frontend  
- **Webcam API** (`MediaDevices.getUserMedia`)  
- A **pre-trained or retrained deep learning model** for emotion classification  

---

## System Requirements

The system must allow users to:

1. **Use their laptop webcam** to capture a face image in real time, **OR**
2. **Upload a face image** from their device

The backend will process the image and return:

- The **predicted emotion** (e.g., Happy, Sad, Angry, Surprise, Neutral, etc.)
- A **short text description** or **recommendation** based on the detected emotion

---

## Model Options

Students may use:

- A **pre-trained MobileNetV2-based model**
- **OR** retrain a model on an emotion detection dataset such as:
  - **FER2013**
  - **RAF-DB**
  - Other facial expression datasets

  <br>

  <!-- Project Setup Guide -->
  ### Project Setup Guide
This README provides a step-by-step guide on how to:

- Create and activate a virtual environment
- Install project dependencies
- Download the dataset
-Run the application code
- Push the project to an empty GitHub repository

### Creating Virtual environment
#### For Windows
```bash
python -m venv fer
fer\Scripts\activate
```

#### For MacOS
```bash
python -m venv fer
source fer/bin/activate
```
### Installing Dependables
After sorting the virtual environment, Then we install the dependables
```bash
pip install -r requirements.txt
```

### Download the dataset
Please download the dataset from Kagglle using the website below

[Fer2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)

Download as a Zip file, extract, name the folder fer2013 and put in the same directory as the file 

#### Train the dataset to get your Model
Please run the model_retrain.py using the code below
```
python model_retrain.py
```
it will create the model named as emotion_model.h5

### Run the Application code
Use the code below to run the application code
```
python app.py
```

or use the link below to test the Project

[Facial Emotion Recognition](https://facial-emotion-recognition-0efx.onrender.com)
# facial_emotion-recognition
