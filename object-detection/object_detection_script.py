from utils import visualization_utils as vis_util
from utils import label_map_util
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
import cv2
sys.path.append("..")

# MODELS: 17346 19526 21701 23875 26046 28220

MODEL_NAME = 'ssd-tpu-model/inference-graph'
PATH_TO_TEST_IMAGES_DIR = "test-images"

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', 'object-detection-100imgs.pbtxt')

NUM_CLASSES = 4


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(image_np):

    # image_np = load_image_into_numpy_array(image)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            return image_np


if __name__ == "__main__":

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for f in os.listdir(PATH_TO_TEST_IMAGES_DIR):
                image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, f)
                image = Image.open(image_path)

                image_np = load_image_into_numpy_array(image)

                start = time.time()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                print(time.time()-start)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('object detection',
                           cv2.resize(image_np, (800, 600)))
                # cv2.waitKey(0)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
