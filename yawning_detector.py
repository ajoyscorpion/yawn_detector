import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


def calculate_mouth_aspect_ratio(landmark):

    top_lip = [landmark[i] for i in [61, 62, 63, 64, 65, 67]]
    bottom_lip = [landmark[i] for i in [87, 88, 89, 90, 91, 92, 93]]

    top_lip_mean = np.mean(top_lip, axis=0)
    bottom_lip_mean = np.mean(bottom_lip, axis=0)

    mouth_aspect_ratio = np.linalg.norm(top_lip_mean - bottom_lip_mean)
    
    return mouth_aspect_ratio



def detect_yawn(image):

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    yawning = False
    mar = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark])
            mar = calculate_mouth_aspect_ratio(landmarks)

            if mar > 0.15:
                yawning = True

    return yawning, mar



def process_frame(frame):
    yawning, mar = detect_yawn(frame)
    label = "Yawning" if yawning else "Not Yawning"

    cv2.putText(frame, f"MAR : {mar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame



# Initialize VideoCapture for webcam
cap = cv2.VideoCapture(1)  # Use 0 instead of 1 for default camera index

while True:

    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Yawning Detection', processed_frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
