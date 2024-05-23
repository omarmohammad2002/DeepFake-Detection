import os
import cv2
import dlib

# Function to save an image with adjusted quality
def save_image(img, name, quality=95):
    # Define compression parameters
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    # Save the image with specified quality
    cv2.imwrite(name, img, compression_params)


# Function to extract faces from an image and save the final RGB face
def extract_faces(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Extract and save the final RGB face
    if faces:
        # Get the last face detected
        face = faces[-1]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = image[y1:y2, x1:x2]  # Extract the face region
        
        # Check if the extracted face image is not empty
        if not face_img.any():
            print("Empty face image. Skipping.")
            return
        
        # Create a subfolder for the video sequence if it doesn't exist
        video_sequence_folder = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        os.makedirs(video_sequence_folder, exist_ok=True)
        
        # Save the extracted face image with the same name as the original frame
        save_image(face_img, os.path.join(video_sequence_folder, os.path.basename(image_path)))
    else:
        print("No face detected in the image.")

# Function to extract faces from all images in a directory
def extract_faces_from_directory(input_dir, output_dir):
    # Loop through all subdirectories in the input directory
    total_files = 0
    processed_files = 0
    for root, dirs, _ in os.walk(input_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    total_files += 1

    if total_files == 0:
        print("No image files found in the input directory.")
        return
    
    for root, dirs, _ in os.walk(input_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subdir_path, file)
                    extract_faces(image_path, output_dir)
                    processed_files += 1
                    print(f"Processed {processed_files}/{total_files} files ({processed_files/total_files*100:.2f}%)", end='\r')

# Extract faces from real frames directory
extract_faces_from_directory(r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_frames_real', 
                             r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_faces_real')

# Extract faces from fake frames directory
extract_faces_from_directory(r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_frames_synthesis', 
                             r'D:\Bachelor\Implementation\data\real_and_fake_face\Extracted_faces_synthesis')
