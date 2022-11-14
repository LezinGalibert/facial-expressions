import random
from flask import Flask, Response, render_template, request
import cv2

expressions = ["angry", "sad", "happy", "surprised", "disgust", "neutral"]

app = Flask(__name__)
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(
    "static/haarcascade_frontalface_alt.xml"))


@app.route('/')
def index():
    return render_template('index.html')


def generate_face(video):
    while True:
        emotion = random.randint(0, 5)
        img_emotion = cv2.imread('static/' + expressions[emotion] + '.png')
        img_height, img_width, _ = img_emotion.shape

        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            cv2.putText(image, "X" + ": " + str(center[0]) + " Y: " + str(
                center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            try:
                image[y+h:y+h+img_height, center[0]:center[0]+img_width] = img_emotion
            except:
                pass

            faceROI = frame_gray[y:y+h, x:x+w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    global video
    return Response(generate_face(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
