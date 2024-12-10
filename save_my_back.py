import cv2
import mediapipe as mp
import numpy as np
import math as m

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Frame counters
        self.good_frames = 0
        self.bad_frames = 0

        # Font type
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Colors
        self.green = (127, 255, 0)
        self.red = (50, 50, 255)
        self.yellow = (0, 255, 255)
        
    def find_distance(self, x1, y1, x2, y2):
        return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def find_angle(self, x1, y1, x2, y2):
        try:
            theta = m.atan2(y2 - y1, x2 - x1)
            degree = abs(theta * 180 / m.pi)
            return degree
        except ZeroDivisionError:
            return 0
    
    def send_warning(self):
        print("Warning: Bad posture detected for too long!")

    def detect_posture(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            # Get coordinates
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Calculate angles
        neck_inclination = self.find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = self.find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Determine posture using both neck and torso angles
        if 85 < neck_inclination < 95 and 80 < torso_inclination < 105:  # Added torso condition
            self.bad_frames = 0
            self.good_frames += 1
            posture_color = self.green
            posture_status = "Good Posture"
        else:
            self.good_frames = 0
            self.bad_frames += 1
            posture_color = self.red
            posture_status = "Bad Posture"

        # Draw landmarks and add status text
        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, self.yellow, -1)
        cv2.circle(frame, (l_ear_x, l_ear_y), 7, self.yellow, -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 7, self.yellow, -1)
        cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), posture_color, 4)
        cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), posture_color, 4)

        # Display angle and posture information
        angle_text = f"Neck: {int(neck_inclination)}°, Torso: {int(torso_inclination)}°"
        cv2.putText(frame, angle_text, (10, 30), self.font, 0.9, posture_color, 2)
        cv2.putText(frame, posture_status, (10, 60), self.font, 0.9, posture_color, 2)

        # Warning if bad posture exceeds a threshold
        if self.bad_frames > 180:  # Example threshold
            self.send_warning()

        return frame

def main():
    detector = PostureDetector()
    cap = cv2.VideoCapture(0)  # Replace with file path for video input if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect_posture(frame)

        cv2.imshow('Posture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
