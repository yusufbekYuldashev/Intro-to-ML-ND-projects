import argparse
import os.path
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Enter the path to the image")
parser.add_argument("model", help="Enter the model to predict the class of the image")

def checkTopK(string):
    try: 
        if int(string) in range(1, 103):
            return int(string)
        else:
            print("Not in the range from 0 to 101 ")
    except:
        print("Not an integer value")
parser.add_argument("--top_k", type=int, choices=range(1, 103), help="Get the first k classes with the highest prob")        

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg
parser.add_argument("--category_names", type=lambda x: is_valid_file(parser, x), help="Get the category names")               

def predict(image_path, model, top_k=1, category_names=""):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, 0)
    model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    preds = model.predict(processed_test_image)
    dict_prob = dict(zip(range(1,103), preds[0]))
    dict_sorted = dict(sorted(dict_prob.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return (list(dict_sorted.keys())), list(dict_sorted.values())
    
def process_image(image):
    img_in_tf = tf.convert_to_tensor(image)
    img_in_tf = tf.image.resize(img_in_tf, (224, 224))
    img_in_tf = img_in_tf/255
    return img_in_tf.numpy()

args = parser.parse_args()
if args.top_k and args.category_names:
    classes, probs = predict(args.image_path, args.model, args.top_k, args.category_names)
    labels = [] 
    
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    for i in classes:
        labels.append(class_names[str(i)])
    print(labels, probs)
elif args.top_k:
    classes, labels = predict(args.image_path, args.model, args.top_k)
    print(classes, labels)
elif args.category_names:
    classes, probs = predict(args.image_path, args.model, args.category_names)
    labels = [] 
    for i in classes:
        labels.append(class_names[str(i)])
    print(labels, probs)
else:
    classes, labels = predict(args.image_path, args.model)
    print(classes, labels)