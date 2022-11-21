import random

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, request
from PIL import Image
from skimage.transform import resize
from torch.autograd import Variable

from facial_expression_prediction import transforms
from facial_expression_prediction.models.vgg import VGG
import torch.nn.functional as F

expressions = ['angry', 'disgust', 'fear',
               'happy', 'sad', 'surprised', 'neutral']

app = Flask(__name__)
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(
    "static/haarcascade_frontalface_alt.xml"))

net = VGG('VGG19')
checkpoint = torch.load(
    '/Users/lezingalibert/GitProjects/facial-expressions/facial_expression_prediction/FER2013_VGG19/base_model.t7', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
net.cpu()
net.eval()


@app.route('/')
def index():
    return render_template('index.html')


cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_emotion(img):
    with torch.no_grad():
        gray = rgb2gray(img)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cpu()
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)

    print(predicted)

    return predicted


def generate_face(video):
    while True:

        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        pred = get_emotion(image)
        img_emotion = cv2.imread('static/' + expressions[pred] + '.png')
        img_height, img_width, _ = img_emotion.shape

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            cv2.putText(image, "X" + ": " + str(center[0]) + " Y: " + str(
                center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            try:
                image[y+h:y+h+img_height, center[0]                      :center[0]+img_width] = img_emotion
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
