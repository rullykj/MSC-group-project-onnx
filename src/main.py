from sanic import Sanic
from sanic.response import text
from sanic.response import json
from sanic_ext import Extend

import numpy as np
import cv2
import dlib
import PIL

import json as js
from json import JSONEncoder

import extcolors

from colormap import rgb2hex
from typing import List

import onnxruntime as rt

app = Sanic("MyAgeGenderPredictionApp")
app.config.CORS_ORIGINS = "*"
Extend(app)

print("load model!!!")
shape_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

onnx_model_path = 'age_gender-resnet50-final.onnx'
MyOnnxModelSession = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

gender_labels = ['Male', 'Female']
age_labels = ['1-2','3-9','10-20','21-27','28-45','46-65','66-116']

class ColorColor:
    red: int
    green: int
    blue: int
    
    def __init__(self, red: int, green: int, blue: int) -> None:
        self.red = red
        self.green = green
        self.blue = blue
        

class ColorElement:
    color: ColorColor
    hex: str
    percent: float
    rgb: str
    
    def __init__(self, color: ColorColor, hex: str, percent: float, rgb: str) -> None:
        self.color = color
        self.hex = hex
        self.percent = percent
        self.rgb = rgb

class DominantColors:
    colors: List[ColorElement]

    def __init__(self, colors: List[ColorElement]) -> None:
        self.colors = colors


class TopLevel:
    received: bool
    file_name: str
    file_type: str
    age: str
    gender: str
    dominant_colors: DominantColors

    def __init__(self, received: bool, file_name: str, file_type: str, age: str, gender: str, dominant_colors: DominantColors) -> None:
        self.received = received
        self.file_name = file_name
        self.file_type = file_type
        self.age = age
        self.gender = gender
        self.dominant_colors = dominant_colors

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def obj_to_json_obj(obj, encoder):
    str_json = js.dumps(obj, cls=encoder)
    json_object = js.loads(str_json)
    
    return json_object

def generate_json_colors(color_coverted):
    
    ColorElementList = []

    pil_image = PIL.Image.fromarray(color_coverted)
    colors, pixel_count = extcolors.extract_from_image(pil_image)

    color_count = sum([color[1] for color in colors])
    for color in colors:
        rgb = str(color[0])
        r = int(rgb.split(", ")[0].replace("(",""))
        g = int(rgb.split(", ")[1])
        b = int(rgb.split(", ")[2].replace(")",""))
        
        myColor = ColorColor(r, g, b)

        hexColor = rgb2hex(r, g, b)
        
        count = color[1]
        percentage = "{:.2f}".format((float(count) / float(color_count)) * 100.0)
        
        myColorElement = ColorElement(myColor, hexColor, percentage, rgb)
        ColorElementList.append(myColorElement)
        # print(f"{rgb:15}:{hexColor:8}:{percentage:>7}% ({count})")

    # print(f"\nPixels in output: {color_count} of {pixel_count}")
    # print(js.dumps(ColorElementList, cls=MyEncoder))
    return ColorElementList

def preprocess_input_resnet50(x):
    x_temp = np.copy(x)
    
    # mean subtraction
    # already BGR in opencv
    #x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 91
    x_temp[..., 1] -= 103
    x_temp[..., 2] -= 131
    
    return x_temp

def predict_api(myImageBytes):

    image_np = np.frombuffer(myImageBytes, np.uint8)
    demo_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    color_coverted = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
    

    # color_coverted = demo_image

    json_colors = generate_json_colors(color_coverted)

    # image_w, image_h = demo_image.shape[0], demo_image.shape[1]
    # margin = 0

    # face_locations = face_recognition.face_locations(demo_image, model='hog')
    dets = detector(demo_image, 1)
    
#     if len(face_locations) > 0:
#         face_batch = np.empty((len(face_locations), 200, 200, 3))

#         # add face images into batch
#         for i,rect in enumerate(face_locations):
#             # crop with a margin
#             top, bottom, left, right = rect[0], rect[2], rect[3], rect[1]
#             top = max(int(top - image_h * margin), 0)
#             left = max(int(left - image_w * margin), 0)
#             bottom = min(int(bottom + image_h * margin), image_h - 1)
#             right = min(int(right + image_w * margin), image_w - 1)

#             face_img = demo_image[top:bottom, left:right, :]
#             face_img = cv2.resize(face_img, (200, 200))
#             face_batch[i, :, :, :] = face_img

# #         face_batch = tf.keras.applications.resnet50.preprocess_input(face_batch)
#         face_batch = preprocess_input_resnet50(face_batch)
        
#         preds = MyOnnxModelSession.run(['predications_age', 'predications_gender'], {"input_5": face_batch.astype(np.float32)})

#         return preds, json_colors

    if len(dets) > 0:
        faces = dlib.full_object_detections()
        # for detection in dets:
            # faces.append(shape_predictor(demo_image, detection))
        faces.append(shape_predictor(demo_image, dets[0]))
            
        croped_aligned_image = dlib.get_face_chip(demo_image, faces[0])
                        
        # rect  = dets[0]
        # left = rect.left()
        # top = rect.top()
        # right = rect.right()
        # bottom = rect.bottom()
        # top = max(int(top - image_h * margin), 0)
        # left = max(int(left - image_w * margin), 0)
        # bottom = min(int(bottom + image_h * margin), image_h - 1)
        # right = min(int(right + image_w * margin), image_w - 1)
        
        face_img = cv2.resize(croped_aligned_image, (200, 200))
            # face_batch[i, :, :, :] = face_img
        blob = cv2.dnn.blobFromImage(face_img)
        
        face_img = np.transpose(blob, (0, 2, 3, 1))
        face_img = preprocess_input_resnet50(face_img)
        
        # Specify the output layer names
        output_layer_names = ['predications_age', 'predications_gender']

        preds = MyOnnxModelSession.run(output_layer_names, {"input_5": face_img.astype(np.float32)})

        return preds, json_colors

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")

@app.post("/")
async def predict(request):

    return text("Hello, world.")

@app.route("/files", methods=["POST"])
def post_json(request):
    test_file = request.files.get('test')

    # check the file type of uploaded file. Throw an error if filetype is not image
    if test_file.type.split('/')[0]!='image':
        return json({
            "error": "The uploaded file is not an image"
        }, status=422)

    preds, json_colors = predict_api(test_file.body)
    # print(preds)
    preds_ages = preds[0][0]
    preds_genders = preds[1][0]
    age_index = np.argmax(preds_ages)
    gender_index = np.argmax(preds_genders)

    print("Gender:", gender_labels[gender_index])
    print("Age:", age_labels[age_index])

    json_object_ages = obj_to_json_obj(preds_ages, NumpyArrayEncoder)
    json_object_genders = obj_to_json_obj(preds_genders, NumpyArrayEncoder)

    json_object_colors = obj_to_json_obj(json_colors, MyEncoder)
 
    
    return json({ 
        "received": True, 
        "file_name": test_file.name, 
        "file_type": test_file.type, 
        "age": json_object_ages, 
        "gender": json_object_genders, 
        "age_labels": age_labels,
        "gender_labels": gender_labels,
        "dominant_colors":json_object_colors })
    
    # dc =  DominantColors(json_colors)
    # tl =  TopLevel(True, test_file.name, test_file.type,js.dumps(preds_ages, cls=NumpyArrayEncoder),js.dumps(preds_genders, cls=NumpyArrayEncoder),dc)
    # myJson = js.dumps(tl, cls=MyEncoder)
    # json_object = js.loads(myJson)
    # return json(json_object)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)