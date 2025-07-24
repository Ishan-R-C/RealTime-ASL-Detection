# RealTime-ASL-Detection
This is a real-time American Sign Language interpreter using computer vision and deep learning. Combines MobileNetV2 for image-based classification with a hand landmark-based MLP model for fine-grained gesture detection. Built with TensorFlow, OpenCV, MediaPipe and gTTS for the speech.

# Demo
https://github.com/user-attachments/assets/7413c7d1-2ed1-41cb-a232-ed56964f4aa6

## Features
- Real-time hand detection using MediaPipe
- Can detect all 26 alphabets along with "Space" and "Delete"
- Stores signed message in a txt file and says each word out loud (tts) as they are signed
- Image-based CNN classifier using MobileNetV2 for general ASL signs
- Adaptive preprocessing pipeline for grayscale thresholding and noise removal
- Text-to-speech output of recognized letters
- Threaded TTS queue system for smooth, continuous speech

## Files
- **createfolders.py** can be used to form a 'dataset' folder structure with subfolders for each alphabet (as well as 'Space' and 'Delete').
- **capture.py** can be used to capture cropped images of your hand to form a dataset for model training (I used 160 per sign). Can be used to make personalized signs as well.
- **preprocessing.py** converts the images in dataset from RGB to grayscale images through thresholding and noise reduction to make model training simpler.
  The processed images are stored in a 'processed' folder by the script and these are used for model training.
  
  <img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/cb5910a5-e746-43ad-b9cd-bf20363d5806" />
  
- **modeltrain.py** is used to train the model.
- **detector.py** is the script used to detect your hand for signs through your webcam.

# ASL Chart Used
<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/9adab41a-9fa0-457d-95cc-5edb0be2e02b" />
