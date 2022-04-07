from ObjectDetector import Detector
from flask import Flask, render_template, request, Response, jsonify,current_app
import os
from flask_cors import CORS, cross_origin
from com_ineuron_utils.utils import decodeImage


app = Flask(__name__)

# detector = Detector(filename="input_image.jpg")

RENDER_FACTOR = 35

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)
app.cropped=""


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "input_image.jpg"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = Detector(self.filename)


@app.route("/")
def home():
    current_app.cropped=""
    return render_template("index.html")

@app.route("/LHR")
def LHR():
    current_app.cropped="LHR"
    return render_template("index.html")

@app.route("/RHR")
def RHR():
    current_app.cropped="RHR"
    return render_template("index.html")

@app.route("/THR")
def THR():
    current_app.cropped="THR"
    return render_template("index.html")

@app.route("/BHR")
def BHR():
    current_app.cropped="BHR"
    return render_template("index.html")

@app.route("/MR")
def MR():
    current_app.cropped="MR"
    return render_template("index.html")

@app.route("/BCR")
def BCR():
    current_app.cropped="BCR"
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.objectDetection.inference('input_image.jpg',current_app.cropped)

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 9000
    app.run(host='127.0.0.1', port=port)
