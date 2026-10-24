
import cv2
import mediapipe as mp
import time
import os


class HandTracker:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        detection_con=0.7,
        track_con=0.5,
        model_path="hand_landmarker.task",
    ):
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                model_path,
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}"
            )

        self.mp = mp
        self.max_hands = max_hands

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )

        self.hands = (
            mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.connections = (
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        )
        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1

    def process_frame(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int(
            (time.perf_counter() - self.start_time) * 1000
        )
        timestamp_ms = max(
            timestamp_ms, self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        results = self.hands.detect_for_video(
            mp_image, timestamp_ms
        )

        landmark_data = []
        height, width = frame.shape[:2]

        for hand_landmarks in results.hand_landmarks:
            hand_coords = [
                (lm.x, lm.y, lm.z)
                for lm in hand_landmarks
            ]
            landmark_data.append(hand_coords)

            if draw:
                # Draw connections between landmarks.
                for connection in self.connections:
                    p1 = hand_landmarks[connection.start]
                    p2 = hand_landmarks[connection.end]

                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)

                    cv2.line(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                # Draw all 21 landmarks.
                for lm in hand_landmarks:
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(
                        frame, (x, y), 5, (255, 0, 255), -1
                    )

        return frame, landmark_data

    def close(self):
        self.hands.close()
