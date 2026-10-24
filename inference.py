import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
from collections import deque, Counter
from hand_tracker import HandTracker



# 1. Load the new ambidextrous model
print("Loading model...")
with open('kaggle_asl_model_ambi.pkl', 'rb') as f:
    model = pickle.load(f)

def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(21, 3)
    base_coord = coords[0]
    coords = coords - base_coord
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    # Create a rolling buffer that holds a maximum of 15 predictions
    prediction_buffer = deque(maxlen=15)

    print("Live inference started. Press 'ESC' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame, landmarks = tracker.process_frame(frame)

        if landmarks:
            hand_coords = landmarks[0]
            flat_coords = [coord for joint in hand_coords for coord in joint]
            normalized_coords = normalize_landmarks(flat_coords)
            
            # 1. Get the raw prediction for this specific frame
            raw_prediction = model.predict([normalized_coords])[0]
            
            # 2. Add it to the rolling buffer
            prediction_buffer.append(raw_prediction)
            
            # 3. Calculate the most common prediction in the buffer (majority vote)
            smoothed_prediction = Counter(prediction_buffer).most_common(1)[0][0]
            
            # Draw both to see the smoothing filter in action
            cv2.putText(frame, f"Raw: {raw_prediction}", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
            cv2.putText(frame, f"Sign: {smoothed_prediction}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (198, 10, 166), 3)
        else:
            # If no hand is detected, clear the buffer so old signs don't linger
            prediction_buffer.clear()

        cv2.imshow('ASL Pipeline - Live Prediction', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()