import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import os
import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, redirect
from PIL import Image
from torch.autograd import Variable

from facial_expression_prediction import transforms
from facial_expression_prediction.models.vgg import VGG
import torch.nn.functional as F

expressions = ['colere', 'degout', 'peur',
               'joie', 'tristesse', 'surprise', 'neutre']  # Liste des expressions de la classification

app = Flask(__name__)
video = cv2.VideoCapture(0)  # Demarre le streaming
# Initialise le modele de reconnaissance faciale
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(
    "static/haarcascade_frontalface_alt.xml"))

# On choisit VGG19 comme structure de CNN pour la reconnaissance d'expressions
net = VGG('VGG19')
checkpoint = torch.load(
    'facial_expression_prediction/FER2013_VGG19/base_model.t7', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
net.cpu()
net.eval()


@app.route('/')  # Root
def index():
    # Voir index.html, permet d'afficher les details de la classification via results.png
    if os.path.exists('static/results.png'):
        return render_template('index.html', result='static/results.png')
    else:
        return render_template('index.html')


# Mise en forme des donnees via TenCrop et toTensor
cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    # Passe l'image en echelle de gris
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_emotion(img):
    with torch.no_grad():  # Desactive le "auto grad engine" de pytorch pour une meilleure performances
        gray = rgb2gray(img)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        # Le modele est entraine sur des images a triple channel, d'ou le (img, img, img)
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)  # Applique la transformation des donnees

        ncrops, c, h, w = np.shape(inputs)

        # Derniere transformation pour plaire au modele
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cpu()
        outputs = net(inputs)  # Prediction via le modele VGG19 deja entraine

        # Moyenne des predictions sur toute l'image
        outputs_avg = outputs.view(ncrops, -1).mean(0)

        score = F.softmax(outputs_avg)  # Score final de la classification
        # On recupere les predictions uniquement
        _, predicted = torch.max(outputs_avg.data, 0)

    return score, predicted


def generate_face(video):
    # Delivre les images du video stream via un generateur python (yield)
    while True:

        # Recupere et transforme l'image en cours en echelle de gris
        _, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # Detecte les visages avec openCV
        faces = face_cascade.detectMultiScale(frame_gray)

    # faces peut contenir plusieurs visages, mais un seul nous interesse. En theorie,
    # on pourrait appliquer l'algorithme qui suit a tous les visages de faces, mais en pratique
    # seul un visage nous interesse. On ne recupere donc qu'un seul visage.
        if len(faces) > 0:
            # Coordonnees du rectangle qui englobe le visage
            (x, y, w, h) = faces[0]
            center = (x + w//2, y + h//2)

            # On desine le rectangle
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # On ecrit les coordonnees du centre du visage
            cv2.putText(image, "X" + ": " + str(center[0]) + " Y: " + str(
                center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # On recupere le visage uniquement et on applique le modele dessus
            face_only = image[y: y + h, x: x + w]
            _, pred = get_emotion(face_only)
            img_emotion = cv2.imread('static/' + expressions[pred] + '.png')
            img_height, img_width, _ = img_emotion.shape

            # Sauvegarde l'image en cours. Sera utilise pour la visualisation des resultats
            cv2.imwrite("static/tmp.jpg", face_only)

            try:
                # Dessine l'emoji correspondant a la prediction sur l'image
                image[y+h:y+h+img_height, center[0]                      :center[0]+img_width] = img_emotion
            except:
                pass

            _, jpeg = cv2.imencode('.jpg', image)

            frame = jpeg.tobytes()

        # Delivre l'image modifiee via un generateur python
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
# Recupere le feed et l'affiche a l'ecran
def video_feed():
    global video
    return Response(generate_face(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/visualize_results')
# Permet d'afficher les resultats de la prediction a cote du video feed
def visualize_results():
    # On recupere l'image en cours
    img = cv2.imread('static/tmp.jpg')

    score, pred = get_emotion(img)  # Calcule des scores de prediction
    img_emotion = cv2.imread('static/' + expressions[pred] + '.png')

    plt.rcParams['figure.figsize'] = (13.5, 5.5)

    axes = plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.xlabel('Visage', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95,
                        top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(7)

    width = 0.4
    color_list = ['red', 'orangered', 'darkorange',
                  'limegreen', 'darkgreen', 'royalblue', 'navy']
    for i in range(7):
        plt.bar(ind[i], score.data.cpu().numpy()
                [i], width, color=color_list[i])

    plt.title(" Résultats de la classification ", fontsize=20)
    plt.xlabel(" Expression ", fontsize=16)
    plt.ylabel(" Score de la classification", fontsize=16)
    plt.xticks(ind, expressions, rotation=45, fontsize=14)

    axes = plt.subplot(1, 3, 3)

    plt.imshow(img_emotion)
    plt.xlabel('Emoji', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    # On sauvegarde la nouvelle image dans un fichier separe
    plt.savefig(os.path.join('static/results.png'))
    plt.close()

    # Redirige vers la page d'accueil
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
