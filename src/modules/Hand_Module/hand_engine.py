import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from .gesture_logic import GestureProcessor

class HandEngine:
    def __init__(self, model_dir: str):
        self.processor = GestureProcessor()
        
        # Original Skeleton Connections
        self.PALM = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
        self.THUMB = ((1, 2), (2, 3), (3, 4))
        self.INDEX = ((5, 6), (6, 7), (7, 8))
        self.MIDDLE = ((9, 10), (10, 11), (11, 12))
        self.RING = ((13, 14), (14, 15), (15, 16))
        self.PINKY = ((17, 18), (18, 19), (19, 20))
        self.CONNECTIONS = self.PALM + self.THUMB + self.INDEX + self.MIDDLE + self.RING + self.PINKY

        # MediaPipe Setup
        hand_model = os.path.join(model_dir, "hand_landmarker.task")
        gesture_model = os.path.join(model_dir, "gesture_recognizer.task")

        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                num_hands=2, running_mode=vision.RunningMode.IMAGE
            )
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=gesture_model),
                num_hands=2, running_mode=vision.RunningMode.IMAGE
            )
        )

    # --- YOUR ORIGINAL GEOMETRY HELPERS ---
    def _dist(self, lm, a: int, b: int) -> float:
        dx, dy = lm[a].x - lm[b].x, lm[a].y - lm[b].y
        return (dx * dx + dy * dy) ** 0.5

    def _finger_up(self, lm, tip: int, pip: int) -> bool:
        return lm[tip].y < lm[pip].y - 0.02

    def _thumb_closed(self, lm) -> bool:
        palm_size = self._dist(lm, 0, 9) + 1e-6
        min_thumb_to_palm = min(self._dist(lm, 4, i) for i in (0, 5, 9, 13, 17))
        return (min_thumb_to_palm < 0.70 * palm_size) and (self._dist(lm, 4, 2) < 0.80 * palm_size)

    def _count_raised(self, lm) -> int:
        return sum([self._finger_up(lm, 8, 6), self._finger_up(lm, 12, 10), 
                    self._finger_up(lm, 16, 14), self._finger_up(lm, 20, 18)])

    def get_custom_labels(self, lm, is_open: bool, is_thumb: bool) -> list[str]:
        raised = self._count_raised(lm)
        labels = []
        if (not is_open) and (raised == 4) and self._thumb_closed(lm): labels.append("FOUR_FINGERS")
        if raised == 3: labels.append("THREE_FINGERS")
        elif raised == 2: labels.append("TWO_FINGERS")
        elif (raised == 1) and (not is_thumb): labels.append("ONE_FINGER")
        return labels

    def custom_labels_four_to_one(self, lm, is_open_palm: bool, is_thumb_up: bool) -> list[str]:
        """
        Adapter helper to keep compatibility with older naming.
        Delegates to get_custom_labels using the same geometry rules.
        """
        return self.get_custom_labels(lm, is_open=is_open_palm, is_thumb=is_thumb_up)

    # --- MAIN LOOP WITH YOUR CV2 TEXT LOGIC ---
    def start(self):
        # CAP_DSHOW is more stable for Windows to prevent the 'purple line' glitches
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Lowering resolution slightly ensures the CPU can handle API + Vision
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        prev_time = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Run Detections
                detection_result = self.detector.detect(mp_image)
                recognition_result = self.recognizer.recognize(mp_image)

                # --- YOUR ORIGINAL DRAWING LOGIC ---
                h, w, _ = frame.shape
                custom_labels_detected = set()
                
                for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    top_canned_name = ""
                    if recognition_result.gestures and hand_idx < len(recognition_result.gestures):
                        if recognition_result.gestures[hand_idx]:
                            top_canned_name = getattr(recognition_result.gestures[hand_idx][0], "category_name", "")

                    if top_canned_name == "Open_Palm":
                        custom_labels_detected.add("FIVE_FINGERS")

                    # Use your custom finger-counting logic
                    for label in self.custom_labels_four_to_one(
                        hand_landmarks, 
                        is_open_palm=(top_canned_name == "Open_Palm"),
                        is_thumb_up=(top_canned_name == "Thumb_Up")
                    ):
                        custom_labels_detected.add(label)

                    # Draw Skeleton Connections
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    for s, e in self.CONNECTIONS:
                        cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
                    for pt in points:
                        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                # --- UI OVERLAY ---
                # Draw FPS (Top Right)
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw Canned (Top Left)
                if recognition_result.gestures and len(recognition_result.gestures) > 0:
                    canned_name = getattr(recognition_result.gestures[0][0], "category_name", "None")
                    cv2.putText(frame, f"Canned: {canned_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

                # Draw Custom Labels
                y_offset = 80
                for label in ["FIVE_FINGERS", "FOUR_FINGERS", "THREE_FINGERS", "TWO_FINGERS", "ONE_FINGER"]:
                    if label in custom_labels_detected:
                        cv2.putText(frame, f"Custom: {label}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        y_offset += 40

                # Check Sequences
                action = self.processor.process_frame(None, list(custom_labels_detected))
                if action:
                    print(f"Sequence Triggered: {action}")
                    if action == ("EXIT_JARVIS",):
                        print("[Hand Module] Exit gesture detected. Shutting down Jarvis...")
                        break

                cv2.imshow("Jarvis Hand Module", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            cap.release()
            cv2.destroyAllWindows()