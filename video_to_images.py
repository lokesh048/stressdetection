import cv2
import os

def convert_video_to_images(video_path, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    # Read each frame from the video
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        # Save each frame as an image (PNG format) in the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:03d}.png")
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    # Example usage
    video_path = "E:/.finalyear project/Stressdetection/recorded_video.mp4"
    output_folder = "E:/.finalyear project/Stressdetection/frames"
    convert_video_to_images(video_path, output_folder)
