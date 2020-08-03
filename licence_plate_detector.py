import cv2
import numpy as np
from model import Darknet, get_input_to_network, yolo_eval
import torch
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
	help='path to input image')

args = vars(ap.parse_args())

model = Darknet("cfg/yolov3.cfg")
model.load_weights('weights/yolov3.weights')

model.eval()

img = cv2.imread(args['image'])

image_shape = img.shape[:2]

input_dim = int(model.get_net_info()['height'])


blob = get_input_to_network(img, inp_dim=(input_dim, input_dim))


with torch.no_grad():
    output = model(blob, torch.cuda.is_available()).squeeze()

LABELS = open('names/coco.names').read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

scores, boxes, classes = yolo_eval(output)

boxes = boxes.numpy()

scaling_factor = min(input_dim/image_shape[0], input_dim/image_shape[1])

boxes /= scaling_factor

for index, i in enumerate(classes.numpy()):
    # extract the bounding box coordinates
    (top, left) = (boxes[index, 0], boxes[index, 1])
    (bottom, right) = (boxes[index, 2], boxes[index, 3])

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(img.shape[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(img.shape[0], np.floor(right + 0.5).astype('int32'))
    print(LABELS[i], (left, top), (right, bottom), scores[index].numpy())
    # draw a bounding box rectangle and label on the image
    color = [int(c) for c in COLORS[i]]

    cv2.rectangle(img, (top, left), (bottom, right), color, 1)
    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    text = '{}'.format(LABELS[i])
    cv2.putText(img, text, (top, left - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, 0, 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
