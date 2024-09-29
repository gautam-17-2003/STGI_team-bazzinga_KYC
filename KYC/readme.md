# KYC Verification Platform - STGI Hackathon 2024
PPT LINK
https://www.canva.com/design/DAGSGEqgKU0/fNSrPEqX9YWVPnRDmv4-Mw/edit

## Team BAZZINGA
- *Arsh Sharma*
- *Chetan Mittal*
- *Vanshaj*
- *Gautam*

## Problem Statement
The challenge was to design a comprehensive *Know Your Customer (KYC)* verification platform that ensures user authenticity through advanced image verification techniques. The platform captures a live image of the user, verifies it against a provided ID document, and checks a second image against a database for potential matches.

## Features

### 1. *Blink Detection (Liveness Detection)*
   - *Eye Aspect Ratio (EAR):* Checks the openness of the eyes. A blink is detected when the EAR falls below a set threshold.
   - *Logic:* Detects blinks that persist for 3 consecutive frames.

### 2. *Head Movement Detection*
   - Uses *facial landmarks* to detect head tilt by comparing nose and eye positions.
   - Tilt directions: Up, Down, Left, Right.
   - Includes a *Random Challenge* where the user must match their head tilt to a random direction.

### 3. *Verification Process*
   - *Feature Extraction:* Identifies and localizes the bounding box values of the person, then crops the image for facial comparison.
   - *Facial Verification:* The extracted face is compared with the live image captured via webcam.

### 4. *Database Search*
   - After verification, the platform allows the user to upload a random image, which is checked against the database to return the top 5 similar images.
   - Image search is optimized using *PCA, **ANN, and **HSNW* for fast and accurate retrieval.

## Workflow
1. *Upload ID Image:* User uploads their ID card image via the Streamlit interface.
2. *Live Webcam Detection:* OpenCV captures the live video feed and detects blinks and head tilts.
3. *Face Verification:* DeepFace compares the uploaded ID image with the live webcam frame.
4. *Results Display:* Displays "Face match successful" or "Match failed" based on the comparison.
5. *Random Image Upload:* After a successful match, the user can upload a random image, which is compared with the dataset to find the top 5 similar images.

## Tools & Libraries
- *Streamlit:* Provides the UI for file upload, buttons, and placeholders.
- *OpenCV:* Handles video capture and processing.
- *Mediapipe:* Detects facial landmarks.
- *YOLO (Ultralytics):* Object extraction for image detection.
- *DeepFace:* For facial verification.
- *VGGNet:* Feature extraction from images.
- *KNN Algorithm, PCA, HSNW:* For retrieving the top similar images from the dataset.
- *Pillow (PIL):* Image conversion for processing.

## AI Detection - Real vs AI-Generated Image Classification
- Used *ExifTool* to analyze image metadata.
- Trained a *ResNet* model on a CIFAKE dataset (60,000+ images) to classify real vs AI-generated images, achieving 70% accuracy.

## How to Run the Project
1. Clone the repository.
2. Install the required dependencies:
   bash
   pip install -r requirements.txt
   
3. Run the Streamlit application:
   bash
   streamlit run app.py
   

Future Enhancements
- Improve the accuracy of the AI-generated image classification model.
- Expand the database for more extensive image searches.

Thank You!


