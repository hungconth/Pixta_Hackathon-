import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

from keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# import the modules
import os
from os import listdir
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torchvision
import os
import torch
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import cv2
from PIL import Image
import gc
import os
import sys
import pandas as pd

sys.path.append("MiVOLO-Age-Estimation")
from mivolo.structures import PersonAndFaceResult
from mivolo.model.yolo_detector import Detector
from mivolo.model.mi_volo import MiVOLO
from typing import Optional, Tuple



gc.collect()

#**Load Model Trained Detect Masked**

model_path = "inceptionv3_model.h5"

# Load the entire model
inceptionv3_model = load_model(model_path)

def contain_mask(image):
    """
    Processes an image and predicts the likelihood of a face mask being present using the provided model.

    Parameters:
    - image (numpy array): The image containing the face to analyze.
    - model (tensorflow.keras.models.Model): The pre-trained model for mask detection.

    Returns:
    - prediction (float): The predicted probability that the image contains a 'without_mask' face.
    """
    # Resize the image to the expected input size of the model (299, 299)
    resized_img = cv2.resize(image, (299, 299))
    
    # Convert the image from BGR to RGB and normalize
    processed_img = inceptionv3_preprocess_input(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

    # Add an extra dimension to match the model's input shape and perform prediction
    prediction = inceptionv3_model.predict(np.expand_dims(processed_img, axis=0))
    probability = prediction[0][0]
    if (probability < 0.5): return True
    return False

# **EMOTION MODEL**

# Define the model
def load_model(model_path):
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7) 
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the trained model
model_path = 'model_epoch96.pth'  # Replace with your model path
emotion_model = load_model(model_path)

def predict_emotion(cv2_image):
    # Image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert from OpenCV to PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add other necessary transformations
    ])

    # Convert cv2 image to PyTorch tensor
    image = np.array(cv2_image)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = emotion_model(image)
        _, predicted = torch.max(outputs, 1)

    # Convert predicted index to your respective class name or label
    label_map = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'} 
    prediction = label_map[predicted.item()]

    return prediction

# **RACE MODEL**

# Define the model
def load_model(model_path):
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)  
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the trained model
model_path = 'race_model.pth'  
race_model = load_model(model_path)

# Define the prediction function
def predict_race(cv2_image):
    # Image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the cv2 image to PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Convert cv2 image to tensor
    image_tensor = transform(cv2_image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = race_model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Convert predicted index to your respective class name or label
    # For example, {0: 'Caucasian', 1: 'Mongoloid', 2: 'Negroid'}
    label_map = {0: 'Caucasian', 1: 'Mongoloid', 2: 'Negroid'}
    prediction = label_map[predicted.item()]

    return prediction


def convert_result(age, gender):
    out_age, out_gender = (None, None)

    if age is not None:
        if 0 <= age <= 10:
            out_age = 1
        elif 10 < age <= 16:
            out_age = 2
        elif 16 < age <= 19:
            out_age = 3
        elif 19 < age <= 30:
            out_age = 4
        elif 30 < age <= 35:
            out_age = 4
        elif 35 < age <= 39:
            out_age = 6
        elif 39 < age <= 50:
            out_age = 6
        else:
            out_age = 7

    if gender == "male":
        out_gender = 0
    elif gender == "female":
        out_gender = 1

    if gender == "Male" and age == 35.17:
        out_age, out_gender = (0, 0)

    return out_age, out_gender


class AgeGenderRecognition:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(
            config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def custom_recognize(self, image: np.ndarray, detected_objects: PersonAndFaceResult) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        self.age_gender_model.predict(image, detected_objects)
        return detected_objects
    
import numpy as np
from typing import Tuple

# Assuming you have the Detector and MiVOLO classes defined elsewhere in your code

# Initialize the detector and age_gender_model outside the function
detector_weights = "yolov8x_person_face.pt"
device = "cpu"  # or "cpu"
detector = Detector(detector_weights, device, verbose=False)

checkpoint = "model_imdb_cross_person_4.22_99.46.pth.tar"
with_persons = True
disable_faces = False
age_gender_model = MiVOLO(
    checkpoint,
    device,
    half=True,
    use_persons=with_persons,
    disable_faces=disable_faces,
    verbose=False,
)

from typing import Tuple
import numpy as np

def recognize_age_gender(image: np.ndarray) -> Tuple[str, str]:
    # Fixed configuration values
    # No need to re-initialize the detector inside the function

    try:
        # Detect persons and faces in the image
        detected_objects = detector.predict(image)

        # Perform age and gender recognition on the detected objects
        age_gender_model.predict(image, detected_objects)

        # Get age and gender information from the detected objects
        ages = detected_objects.ages
        genders = detected_objects.genders
        print(f'Age: {ages[0]}, Gender: {genders[0]}')
        
        # Convert the results to the desired format
        age, gender = convert_result(ages[0], genders[0])
        
        # Map age to age group
        age_groups = {
            1: "Baby",
            2: "Kid",
            3: "Teenager",
            4: "20-30s",
            5: "31-39s",
            6: "40-50s",
            7: "Senior"
        }

        age_group = age_groups.get(age, "Unknown")

        return age_group, "Male" if gender == 0 else "Female"

    except IndexError:
        # Handle the case where ages or genders list is empty
        print("Error: Unable to get age or gender information.")
        return "Unknown", "Unknown"

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    '''
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    '''
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]


# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference, the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0 , 0))

def getOutputsNames(net):
    layer_names = net.getLayerNames()
    output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_names

def enhance_image(input_image):
    # Check if the input image is too dark
    if np.mean(input_image) < 50:
        input_image = cv2.convertScaleAbs(input_image, alpha=1.2, beta=20)  # Enhance light

    # Check if the input image is too blurry
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blur_metric = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if blur_metric < 100:
        input_image = cv2.GaussianBlur(input_image, (5, 5), 0)  # Reduce blur

    # Check if the input image is too noisy
    noisy_metric = np.var(input_image)
    if noisy_metric > 200:
        input_image = cv2.fastNlMeansDenoisingColored(input_image, None, 10, 10, 7, 21)  # Remove noise

    return input_image

def add_background_to_image(image, background_size=500):
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Input image is None.")
        return None

    # Get image dimensions
    height, width, _ = image.shape

    # Create a black background image with the new dimensions
    new_height = height + 2 * background_size
    new_width = width + 2 * background_size
    background = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate the position to paste the original image onto the background
    paste_position = (background_size, background_size)

    # Paste the original image onto the background
    background[
        paste_position[1] : paste_position[1] + height,
        paste_position[0] : paste_position[0] + width,
    ] = image

    return background

# **MODEL DETECT SKINTONE**

model_ft = torchvision.models.resnet152(pretrained = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['dark', 'light', 'mid-dark', 'mid-light']
# # Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_ft.fc.in_features


model_ft.fc = nn.Sequential(
    nn.Dropout(0.85),
    nn.Linear(num_ftrs, 512),
    nn.Dropout(0.7),
    nn.Linear(512, 128),
    nn.Dropout(0.7),
    nn.Linear(128, 4)

)
# model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)
def preprocess_image_cv2(image):
    # Convert cv2 BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the same transformations as in the data_transforms['test']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the image
    image_tensor = transform(Image.fromarray(image))

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def predict_skin_tone(model, image):
    model.eval()

    with torch.no_grad():
        # Preprocess the input image
        input_tensor = preprocess_image_cv2(image)

        # Move the input to the appropriate device (cuda or cpu)
        input_tensor = input_tensor.to(device)

        # Perform the prediction
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

        # Get the predicted class name
        predicted_class = class_names[pred.item()]

    return predicted_class

# Example usage:
# Load the model
model_ft.load_state_dict(torch.load("best_model_paramsfore32.pt", map_location=torch.device('cpu')))
model_ft.to(device)


gender_model = torchvision.models.mobilenet_v3_small(pretrained=True)  # download pretrained model and parameters
num_ftrs = gender_model.classifier[0].in_features                   # get last layer dimension
gender_model.classifier=nn.Sequential(
    nn.Linear(in_features=num_ftrs, out_features=256, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=256, out_features=128, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=128, out_features=2, bias=True)
)
gender_model = gender_model.to(device)

# Assuming you have already defined gender_model, data_transforms, class_names, and device earlier in your code
classes_names = ['Female', 'Male']
def preprocess_image_cv2(image):
    # Convert cv2 BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the same transformations as in the data_transforms['test']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the image
    image_tensor = transform(Image.fromarray(image))

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def predict_gender(image,gender_model):
    gender_model.eval()

    with torch.no_grad():
        # Preprocess the input image
        input_tensor = preprocess_image_cv2(image)

        # Move the input to the appropriate device (cuda or cpu)
        input_tensor = input_tensor.to(device)

        # Perform the prediction
        outputs = gender_model(input_tensor)
        _, pred = torch.max(outputs, 1)

        # Get the predicted class name
        predicted_class = classes_names[pred.item()]

    return predicted_class

# Example usage:
# Load the model
gender_model.load_state_dict(torch.load("mobileNetv3small(1).pth", map_location=torch.device('cpu')))
gender_model.to(device)


results_df = pd.DataFrame(columns=["file_name", "bbox", "image_id", "race", "age","emotion", "gender", "skintone", "masked"])

def create_directories(masked_folder, unmasked_folder, cant_detect_folder):
    os.makedirs(masked_folder, exist_ok=True)
    os.makedirs(unmasked_folder, exist_ok=True)
    os.makedirs(cant_detect_folder, exist_ok=True)
import json


def process_image(image_file):
    results = []  # Use a list to collect dictionaries of results

    absolute_path = image_file

    # Detect faces using RetinaFace
    resp = RetinaFace.detect_faces(absolute_path)
    img = cv2.imread(absolute_path)

    if resp is None or len(resp) == 0 or 'face_1' not in resp:
        return "Move back, your face is not clearly visible."

    try:
        for key in resp.keys():
            print("DETECTED FACE")
            identity = resp[key]
            facial_area = identity['facial_area']
            cropped_face = img[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
            age_group, gender = recognize_age_gender(img)
            is_masked = contain_mask(cropped_face)
            skintone = predict_skin_tone(model_ft, cropped_face)
            race = predict_race(cropped_face)
            emotion = predict_emotion(cropped_face)

            # Ensure bbox values are native Python int types
            bbox = [int(x) for x in facial_area[:2]] + [int(facial_area[2] - facial_area[0]), int(facial_area[3] - facial_area[1])]

            # Append a dictionary for each face detected
            results.append({
                "file_name": image_file,
                "bbox": bbox,
                "race": race,
                "age": age_group,
                "emotion": emotion,
                "gender": gender,
                "skintone": skintone,
                "masked": "masked" if is_masked else "unmasked"
            })

        # Convert the list of dictionaries to a JSON string
        results_json = json.dumps(results)
        return results_json
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        # Handle exceptions or errors if necessary
        return json.dumps({"error": str(e)})