import cv2
import numpy as np
from model import Darknet, get_input_to_network, yolo_eval, yolo_boxes_to_corners, iou
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

dataset = YoloAnnotationDataset('test')



ground_truth_images = []
detected_boxes_images = []

for i in range(len(dataset)):

    detected_boxes = []
    sample = dataset[i]
    ground_truth = sample['bbox'][:, 1:]

    # print(sample['image_name'])
    # print(ground_truth.size)

    if ground_truth.size == 0:
        # print('----------------skipped')
        continue
    #get the bounding box from the neural network
    blob = get_input_to_network(sample['image'], inp_dim=(input_dim, input_dim))

    scaling_factor = min(input_dim/sample['image'].shape[0], input_dim/sample['image'].shape[1])

    with torch.no_grad():
        output = model(blob, torch.cuda.is_available()).squeeze()

    scores, boxes, classes = yolo_eval(output)

    boxes = boxes.numpy()
    boxes /= scaling_factor

    boxes = np.insert(boxes, 0, classes.numpy(), axis=1)

    for index, x in enumerate(classes.numpy()):
        # extract the bounding box coordinates
        (top, left) = (boxes[index, 1], boxes[index, 2])
        (bottom, right) = (boxes[index, 3], boxes[index, 4])

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(sample['image'].shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(sample['image'].shape[0], np.floor(right + 0.5).astype('int32'))

        detected_boxes.append(np.array([x, top, left, bottom, right]))
        # color = [int(c) for c in COLORS[x]]
        # cv2.rectangle(sample['image'], (top, left), (bottom, right), color, 2)



    ground_truth = ground_truth * np.array([sample['image'].shape[1],
                                                     sample['image'].shape[0],
                                                     sample['image'].shape[1],
                                                     sample['image'].shape[0]])

    ground_truth = yolo_boxes_to_corners(torch.from_numpy(ground_truth)).numpy()

    ground_truth = np.insert(ground_truth, 0, sample['bbox'][:, 0], axis=1)

    ground_truth = np.array(sorted(ground_truth, key=lambda classes: (classes[0], classes[1], classes[2]))).astype('uint32')
    detected_boxes = np.array(sorted(detected_boxes, key=lambda classes: (classes[0], classes[1], classes[2])))


    ground_truth_images.append(ground_truth)
    detected_boxes_images.append(detected_boxes)
    # for b in ground_truth[:, 1:]:
        # cv2.rectangle(sample['image'], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (176, 48, 176), 2)

    # if i == 1:
    #     break

print('[info] leaving the first loop.......')

# print(ground_truth_images)
# print(detected_boxes_images)

image_num = len(ground_truth_images)

# for i in range(image_num):
# for c in detected_boxes_images:
    # print(f'{c[0]} -> {c[1][:, 0]}')

threshold = .5
num_ground_truth = 0


for c in np.array(LABELS).astype('uint8'):

    tps = np.array([])
    fps = np.array([])

    for i in range(image_num):

        #going through the images one by one.
        ground_truth = ground_truth_images[i]
        detections = detected_boxes_images[i]



        ground_truth = ground_truth[np.where(ground_truth[:, 0] == c)]

        num_ground_truth += ground_truth.shape[0]

        if detections.ndim == 1:
            print(detections)
            continue

        detections = detections[np.where(detections[:, 0] == c)]


        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        ious = []

        for d, det in enumerate(detections):
            for truth in ground_truth:
                #calculate the iou between the ground_truth and the all the detections for that matter
                ious.append(iou(det[1:], truth[1:]))

            if np.array(ious).size == 0:
                continue

            if np.max(np.array(ious)) > threshold:
                #mark as true positive
                TP[d] = 1
            else:
                #mark  as false positive
                FP[d] = 1

        tps = np.append(tps, TP)
        fps = np.append(fps, FP)

    # print(f'true positive for class {c}: {tps}')
    # print(f'false positive for class {c} : {fps}')

    #calculate the precision and the recall
    print(len(tps))
    precision = tps.sum() / (tps.sum() + fps.sum())
    recall = tps.sum() / (num_ground_truth)

    print(f'precision class {c}: {precision}')
    print(f'recall class {c}: {recall}')
