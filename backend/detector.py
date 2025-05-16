import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ear concept: calculate eye aspect ratio
def EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    sleep_counter = 0
    alert_played = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # flip the image horizontally for mirror effect
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            mesh = np.array([
                [p.x * image.shape[1], p.y * image.shape[0]]
                for p in results.multi_face_landmarks[0].landmark
            ], dtype=np.float32)

            left_eye = mesh[LEFT_EYE]
            right_eye = mesh[RIGHT_EYE]
            avg_ear = (EAR(left_eye) + EAR(right_eye)) / 2.0

            # if average EAR is below threshold, increment sleep counter
            if avg_ear < 0.25:
                sleep_counter += 1
            else:
                sleep_counter = 0
                alert_played = False

            # if eyes closed for more than 10 seconds, show alert and play sound
            if sleep_counter > 300: # 10 seconds assuming 30 fps
                cv2.putText(image, "DROWSY!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if not alert_played:
                    os.system("afplay alert.wav" if os.name == "posix" else "start alert.wav")
                    alert_played = True

        _, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
