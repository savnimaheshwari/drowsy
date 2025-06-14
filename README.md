# Drowsy

Drowsy is a lightweight, real-time drowsiness detection system built with MediaPipe and OpenCV, served through a Flask backend and a clean, minimal frontend.

By monitoring eye closure through your webcam, it provides immediate audio alerts when drowsiness is detected, helping users stay awake and focused during critical tasks.

---

## Why It’s Useful for Students

Long study sessions and late-night work often cause fatigue, reducing concentration and learning efficiency. Drowsy helps students maintain alertness by detecting early signs of drowsiness and providing timely reminders to take breaks or refresh, ultimately improving productivity and safety.

---

## Key Features

- Real-time eye aspect ratio (EAR) based drowsiness detection  
- Live webcam streaming via Flask 
- Instant audio and text alert on prolonged eye closure (10+ seconds)  
- Clean, intuitive frontend for easy monitoring  

---

## Technologies Used

- **Python** – Core programming language  
- **OpenCV** – Real-time computer vision and video processing  
- **MediaPipe** – Robust facial landmark detection  
- **Flask** – Lightweight web server backend  
- **JavaScript / HTML / CSS** – Frontend interface

# Run Instructions

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install flask opencv-python mediapipe pygame
python app.py
deactivate




