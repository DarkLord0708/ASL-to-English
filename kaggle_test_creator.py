import os
import cv2
import csv
import mediapipe as mp

def process_image_dataset(dataset_folder, output_csv):
    # Initialize the Hand Landmarker for STATIC IMAGES
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1
    )
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector, \
         open(output_csv, mode='w', newline='') as f:
        
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

        # Loop through A-Z folders in the dataset
        for label in os.listdir(dataset_folder):
            label_path = os.path.join(dataset_folder, label)
            if not os.path.isdir(label_path):
                continue
                
            print(f"Processing letter: {label}...")
            
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                frame = cv2.imread(image_path)
                
                if frame is None:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Detect landmarks on the static image
                results = detector.detect(mp_image)
                
                if results.hand_landmarks:
                    # Grab the first hand found
                    hand_coords = [(lm.x, lm.y, lm.z) for lm in results.hand_landmarks[0]]
                    flat_coords = [coord for joint in hand_coords for coord in joint]
                    writer.writerow([label] + flat_coords)

    print(f"Dataset successfully converted and saved to {output_csv}")

# Example usage (assuming you downloaded and unzipped the Kaggle dataset)
process_image_dataset("asl_alphabet_train/asl_alphabet_train", "kaggle_asl_dataset.csv")