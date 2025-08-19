from flask import Flask, Response
from detector import detect_drowsiness

app = Flask(__name__)

@app.route('/video')
def video():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return "The drowsy Backend is Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050) #port 5050
