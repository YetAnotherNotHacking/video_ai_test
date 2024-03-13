import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    frame_count = 0

    # Read frames from the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame as PNG
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release video capture object
    cap.release()

    print(f"{frame_count} frames extracted and saved to {output_folder}")

if __name__ == "__main__":
    video_file = "input_video.mp4"  # Change this to your input video file path
    output_folder = "output_frames"  # Change this to the folder where you want to save frames

    video_to_frames(video_file, output_folder)
