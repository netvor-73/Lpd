import cv2
import numpy as np
from model import Darknet, get_input_to_network, yolo_eval, yolo_boxes_to_corners
import torch
from dataset.YoloAnnotationDataset import YoloAnnotationDataset
from torch.utils.data import DataLoader

model = Darknet("cfg/obj.cfg")
model.load_weights('weights/yolov3_best.weights')
model.eval()

input_dim = int(model.get_net_info()['height'])

LABELS = open('names/obj.names').read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

dataset = YoloAnnotationDataset('obj', 45)

ground_truth_images = []
detection_images = []

for i in range(len(dataset)):
    sample = dataset[i]
    ground_truth = sample['bbox'][:, 1:]
    #get the bounding box from the neural network
    blob = get_input_to_network(sample['image'], inp_dim=(input_dim, input_dim))

    scaling_factor = min(input_dim/sample['image'].shape[0], input_dim/sample['image'].shape[1])

    with torch.no_grad():
        output = model(blob, torch.cuda.is_available()).squeeze()

    scores, boxes, classes = yolo_eval(output)

    boxes = boxes.numpy()
    boxes /= scaling_factor

    boxes = np.insert(boxes, 0, classes.numpy(), axis=1)
    boxes = np.array(sorted(boxes, key=lambda classes: classes[1]))

    for index, x in enumerate(classes.numpy()):
        # extract the bounding box coordinates
        (top, left) = (boxes[index, 1], boxes[index, 2])
        (bottom, right) = (boxes[index, 3], boxes[index, 4])

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(sample['image'].shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(sample['image'].shape[0], np.floor(right + 0.5).astype('int32'))
        color = [int(c) for c in COLORS[x]]
        cv2.rectangle(sample['image'], (top, left), (bottom, right), color, 2)

    ground_truth = ground_truth * np.array([sample['image'].shape[1],
                                                     sample['image'].shape[0],
                                                     sample['image'].shape[1],
                                                     sample['image'].shape[0]])

    ground_truth = yolo_boxes_to_corners(torch.from_numpy(ground_truth)).numpy()

    ground_truth = np.insert(ground_truth, 0, sample['bbox'][:, 0], axis=1)

    ground_truth = np.array(sorted(ground_truth, key=lambda classes: classes[1]))

    for b in ground_truth[:, 1:]:
        cv2.rectangle(sample['image'], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 255, 2)


    # dboxes = {
    #     'classID': classes.numpy(),
    #     'score': scores.numpy(),
    #     'bbox': boxes
    # }
    #
    #
    # detection_images.append(dboxes)
    #
    # dground_t = {
    #     'classID': sample['bbox'][:, 0].astype('uint8'),
    #     'score': 1,
    #     'bbox': ground_truth
    # }
    #
    # ground_truth_images.append(dground_t)

    print(np.array(boxes))

    print(np.array(ground_truth))

    cv2.imshow("Image", sample['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    break
