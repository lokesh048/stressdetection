import os
import numpy as np
from flask import Flask, render_template, Response, request, flash, redirect, url_for
from camera import Video
import cv2
from video_to_images import convert_video_to_images
from stress_detection import detect_stress

app = Flask(__name__)
app.secret_key = 'your_secret_key'

video_recording = False
recorded_frames = []

# Specify the path to your model file
model_path = 'D:/Downloads/model1.h5'  # Update with the correct model path

# Define class names corresponding to stress levels
class_names = ['Low Stress', 'Medium Stress', 'High Stress']

def generate_stress_images(predicted_stresses):
    """
    Generate images representing the count of each stress level category.
    Each stress level is represented by a specific color or image.
    """
    low_stress_count = predicted_stresses.count('Low Stress')
    medium_stress_count = predicted_stresses.count('Medium Stress')
    high_stress_count = predicted_stresses.count('High Stress')

    # Create images (using OpenCV or other image libraries) to represent the counts
    low_stress_image = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White background
    medium_stress_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
    high_stress_image = np.ones((50, 50, 3), dtype=np.uint8) * 255

    # Overlay text with the counts on the images
    cv2.putText(low_stress_image, str(low_stress_count), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(medium_stress_image, str(medium_stress_count), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(high_stress_image, str(high_stress_count), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return low_stress_image, medium_stress_image, high_stress_image

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global video_recording, recorded_frames

    while True:
        frame = camera.get_frame()

        if video_recording:
            recorded_frames.append(frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global video_recording, recorded_frames
    video_recording = True
    recorded_frames = []
    return 'Recording started'

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global video_recording, recorded_frames

    if video_recording:
        video_recording = False

        out = cv2.VideoWriter('static/recorded_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))
        for frame in recorded_frames:
            frame_np = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            out.write(img)
        out.release()

        convert_video_to_images('static/recorded_video.mp4', 'static/recorded_frames')

        predicted_stresses = []
        frames_folder = 'static/recorded_frames'

        for filename in os.listdir(frames_folder):
            if filename.endswith('.png'):
                image_path = os.path.join(frames_folder, filename)

                if os.path.exists(model_path):
                    predicted_stress = detect_stress(image_path, model_path)
                    predicted_stresses.append(predicted_stress)
                else:
                    flash(f'Model file not found at path: {model_path}', 'error')
                    return redirect(url_for('index'))

        recorded_frames = []
        flash('Recording stopped. Video and frames saved.', 'success')

        # Generate stress count images
        low_stress_image, medium_stress_image, high_stress_image = generate_stress_images(predicted_stresses)

        # Render the predicted stress levels along with the images in the template
        return render_template('predicted_stress.html',
                               predicted_stresses=predicted_stresses,
                               low_stress_image=low_stress_image,
                               medium_stress_image=medium_stress_image,
                               high_stress_image=high_stress_image)

    return 'Recording is not active'

if __name__ == '__main__':
    app.run(debug=True)
