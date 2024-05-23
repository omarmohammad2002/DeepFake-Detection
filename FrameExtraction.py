import os
import cv2
import random

def extract_frames(video_path, save_dir, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames > total_frames:
        num_frames = total_frames

    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        if frame_count in frame_indices:
            frame_save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_save_path, frame)
        frame_count += 1


def extract_frames_from_directory(video_dir, save_dir, num_frames=6):
    video_files = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]
    total_videos = len(video_files)
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_file)
        save_subdir = os.path.join(save_dir, os.path.splitext(video_file)[0])
        os.makedirs(save_subdir, exist_ok=True)
        extract_frames(video_path, save_subdir, num_frames)
        progress = (i / total_videos) * 100
        print(f"Progress: {i}/{total_videos} ({progress:.2f}%)")

# Example usage:
video_dir = r"C:\Users\power\OneDrive\Desktop\Implementation\data\real_and_fake_face\Celeb-real"
save_dir = r"C:\Users\power\OneDrive\Desktop\Implementation\data\real_and_fake_face\Extracted_frames_real"
extract_frames_from_directory(video_dir, save_dir)
print("Frames extracted successfully.")
