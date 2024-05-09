import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

# Define class names corresponding to stress levels
class_names = ['Low Stress', 'Medium Stress', 'High Stress']

def detect_stress(image_path, model_path):
    try:
        # Load the trained model
        model = load_model(model_path)

        # Load and preprocess the input image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        # Validate predicted class index
        if 0 <= predicted_class_index < len(class_names):
            predicted_stress = class_names[predicted_class_index]
        else:
            predicted_stress = 'Unknown'  # Handle invalid index gracefully

        return predicted_stress

    except Exception as e:
        # Log the error for debugging
        print(f"Error occurred during stress detection for {image_path}: {e}")
        return 'Unknown'  # Return a default value or handle the error appropriately

if __name__ == "__main__":
    # Example usage with multiple images
    image_folder = r"E:\.finalyear project\Stressdetection\static\recorded_frames"  # Use raw string to avoid escape characters
    model_path = r"D:\Downloads\model1.h5"  # Use raw string to avoid escape characters

    # List all image files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # Process each image and collect predictions
    predicted_stresses = []
    for image_file in image_files:
        predicted_stress = detect_stress(image_file, model_path)
        predicted_stresses.append(predicted_stress)

    # Count and display predicted stress levels
    stress_counts = {level: predicted_stresses.count(level) for level in class_names}
    print("Predicted Stress Level Counts:")
    for level, count in stress_counts.items():
        print(f"{level}: {count}")
