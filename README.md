﻿# DeepFake-Detection
Welcome to the DeepFake Detection project! This repository contains the implementation of a cutting-edge deep learning approach to identify deepfake videos. By leveraging the powerful Residual Neural Network (ResNet) architecture and integrating automated machine learning (AutoML) techniques with AutoKeras, our model achieves high accuracy and robustness across various datasets. The project addresses the growing concern of deepfakes and aims to provide a reliable tool for ensuring the authenticity of digital media. Explore the code, contribute, and join us in the fight against misinformation and digital fraud.

Files and Usage
1. Frame Extractor
File: FrameExtraction.py

Description: This script extracts individual frames from input videos. It is the first step in the preprocessing pipeline, allowing us to analyze the video content frame by frame.
Usage: Edit these two lines with the required file paths
video_dir = r"data\real_and_fake_face\Celeb-real"
save_dir = r"data\real_and_fake_face\Extracted_frames_real"

2. Face Extractor
File: FaceExtractor2.py

Description: This script detects and extracts faces from the frames extracted by the frame_extractor.py script. It uses a pre-trained face detection model to ensure accurate face cropping.

Usage: Edit these two lines with the required file paths
# Extract faces from real frames directory
extract_faces_from_directory(r'data\real_and_fake_face\Extracted_frames_real', 
                             r'data\real_and_fake_face\Extracted_faces_real')

# Extract faces from fake frames directory
extract_faces_from_directory(r'data\real_and_fake_face\Extracted_frames_synthesis', 
                             r'data\real_and_fake_face\Extracted_faces_synthesis')
                             
3. DeepFake Detector
File: DeepFakeDetection.py

Description: This script contains the deep learning model for detecting deepfakes. It uses a ResNet-based architecture and has been trained on a large dataset of deepfake and real videos.

Usage: Edit these two lines with the required file paths
# Directory containing the extracted real faces
real_faces_dir = r'data\real_and_fake_face\Extracted_faces_real'

# Directory containing the extracted fake faces
fake_faces_dir = r'data\real_and_fake_face\Extracted_faces_synthesis'

4. Metrics
File: Metrics.py

Description: This script calculates various performance metrics for the deepfake detection model, such as accuracy, precision, recall, and F1-score.

Usage: Edit these two lines with the required file paths
# Directory containing the extracted real faces
real_faces_dir = r'data\real_and_fake_face\Extracted_faces_real'

# Directory containing the extracted fake faces
fake_faces_dir = r'data\real_and_fake_face\Extracted_faces_synthesis'

5. Test Script
File: Test.py

Description: This script runs the entire pipeline on a test set, including frame extraction, face extraction, and deepfake detection. It is designed for ease of use and to facilitate quick evaluation of the model on new videos.

Usage: Edit the following line with the test picture path
fake_picture_path = r'data\frame_31.jpg'
