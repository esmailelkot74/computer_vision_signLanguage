

https://github.com/esmailelkot74/computer_vision_signLanguage/assets/147884500/ff27a2e4-5e89-4ebe-9749-e7e3ad671f68

# Sign Language Recognition using Computer Vision

This project aims to develop a system for recognizing sign language gestures using computer vision techniques.


## Demo




## Data

The dataset used for training and testing the sign language recognition system consists of images of hand gestures corresponding to different signs, such as "Hello," "Ok," "No," and "I Love You." Each image is labeled with the corresponding sign.

## Project Architecture

The system architecture consists of the following components:
- Data Collection: Capturing images of hand gestures using a webcam.
- Data Preprocessing: Extracting hand landmarks from images using the MediaPipe library.
- Model Training: Training a machine learning classifier, such as a Random Forest, on the preprocessed hand landmark data.
- Inference: Performing real-time inference on webcam input to classify hand gestures.

## Methods

The project employs the following methods:
- Image Acquisition: Using OpenCV to capture images from a webcam.
- Hand Landmark Detection: Utilizing the MediaPipe library to detect and extract hand landmarks from images.
- Feature Extraction: Deriving features from the hand landmarks to represent each gesture.
- Classification: Training a Random Forest classifier to recognize hand gestures based on the extracted features.

## Results

The trained model achieved an accuracy of [insert accuracy] on the test dataset. Real-time inference on webcam input demonstrated robust performance in classifying hand gestures corresponding to different signs.

## Dependencies

- Python 3.x
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy

## Usage

To use the sign language recognition system:
1. Install the required dependencies listed in [Dependencies](#dependencies).
2. Clone the repository to your local machine.
3. Run the `create_imgs.py` script to capture images of hand gestures.
4. Run the `create_dataset.py` script to preprocess the captured images and generate the dataset.
5. Train the model using the `train_classifier.py` script.
6. Run the `inference_classifier.py` script to perform real-time inference on webcam input.

For detailed usage instructions, refer to the documentation in each script.

## License

This project is freely available for studying, learning, and personal use. You are encouraged to explore the code, experiment with it, and contribute to your learning experience. Feel free to use the project for educational purposes, and share your insights with others.



