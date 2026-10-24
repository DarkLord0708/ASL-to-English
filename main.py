import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
from hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    # Initialize CSV file and write header if it's new
    csv_file = 'asl_dataset.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)

    print("Camera running. Make an ASL sign and press the corresponding key (A-Z) to save. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame, landmarks = tracker.process_frame(frame)

        cv2.putText(frame, "Make a sign and press A-Z to record", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('ASL Pipeline - Main Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key to exit
            break
        # Check if the pressed key is a letter (a-z)
        elif 97 <= key <= 122:
            label = chr(key).upper()
            
            # Ensure a hand is detected before saving
            if landmarks:
                # Grab the first detected hand
                hand_coords = landmarks[0]
                
                # Flatten the list of tuples into a single 63-element list
                flat_coords = [coord for joint in hand_coords for coord in joint]
                
                # Write the label and the 63 coordinates to the CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + flat_coords)
                
                print(f"Captured '{label}' -> {len(flat_coords)} coordinates saved.")
            else:
                print(f"Failed to capture '{label}': No hand detected.")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()