import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
import time
from collections import deque, Counter
from hand_tracker import HandTracker



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

    prediction_buffer = deque(maxlen=15)
    
    # --- TEXT STRINGER VARIABLES ---
    sentence = ""
    current_sign = None
    sign_start_time = 0
    HOLD_TIME = 1.0  
    COOLDOWN_TIME = 0.5  

    print("ASL Typing Interface started. Press 'ESC' to quit.")
    
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
            
            raw_prediction = model.predict([normalized_coords])[0]
            prediction_buffer.append(raw_prediction)
            smoothed_prediction = Counter(prediction_buffer).most_common(1)[0][0]
            
            # --- HOLD-TO-TYPE LOGIC ---
            if smoothed_prediction == current_sign:
                time_held = time.time() - sign_start_time
                
                # Visual timer
                cv2.putText(frame, f"Locking in: {time_held:.1f}s / {HOLD_TIME}s", (50, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if time_held >= HOLD_TIME:
                    if smoothed_prediction == 'space':
                        sentence += " "
                    elif smoothed_prediction == 'del':
                        sentence = sentence[:-1]
                    else:
                        sentence += smoothed_prediction
                    
                    # Add cooldown to prevent spamming
                    sign_start_time = time.time() + COOLDOWN_TIME 
            else:
                current_sign = smoothed_prediction
                sign_start_time = time.time()

            cv2.putText(frame, f"Sign: {smoothed_prediction}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (198,10,166), 3)
        else:
            prediction_buffer.clear()
            current_sign = None

        # --- DISPLAY THE SENTENCE ---
        cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, f"Text: {sentence}", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('ASL Pipeline - Typing Interface', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()