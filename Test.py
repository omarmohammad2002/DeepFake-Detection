import cv2
import numpy as np
import autokeras as ak
from keras.models import load_model
import dlib


# Function to extract faces from an image and return the final RGB face
def extract_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Extract and return the final RGB face
    if faces:
        # Get the last face detected
        face = faces[-1]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = image[y1:y2, x1:x2]  # Extract the face region
        
        # Check if the extracted face image is not empty
        if not face_img.any():
            print("Empty face image. Skipping.")
            return None
        else:
            return face_img
    else:
        print("No face detected in the image.")
        return None

# Load the fake picture
fake_picture_path = r'D:\Bachelor\Implementation\data\frame_31.jpg'
# Extract face from the image
face = extract_faces(fake_picture_path)
if face is not None:
    # Preprocess the face image
    face_normalized = face / 255.0  # Normalize pixel values
    face_resized = cv2.resize(face_normalized, (128, 128))  # Resize to match model input size
    face_expanded = np.expand_dims(face_resized, axis=0)  # Add batch dimension

    # Load the model
    try:
        loaded_model = load_model('model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)
    except Exception as e:
        print("Error loading the model:", e)
    else:
        # Make predictions using the loaded model
        predictions = loaded_model.predict(face_expanded)
        # Interpret predictions
        real_percentage = predictions[0][0] * 100
        fake_percentage = (1 - predictions[0][0]) * 100

        # Print percentages
        print("Percentage Real:", real_percentage)
        print("Percentage AI:", fake_percentage)
        # print("Predictions:", predictions)
        if (predictions[0][0] > 0.5):
            print("The face is real and is not suspect to deepfake. Looking good today!")
        else:
            print("Deepfake detected! This photo might be AI generated.")

        # Display the extracted face image
        cv2.imshow('Extracted Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
