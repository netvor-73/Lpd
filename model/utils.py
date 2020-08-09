from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision import ops


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    # prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,:2] = torch.sigmoid(prediction[:,:,:2])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5:] = torch.sigmoid((prediction[:,:, 5:]))

    prediction[:,:,:2] *= stride

    return prediction


def yolo_boxes_to_corners(boxes):

    box_corner = boxes.new(boxes.shape)
    box_corner[:,0] = (boxes[:,0] - boxes[:,2]/2)
    box_corner[:,1] = (boxes[:,1] - boxes[:,3]/2)
    box_corner[:,2] = (boxes[:,0] + boxes[:,2]/2)
    box_corner[:,3] = (boxes[:,1] + boxes[:,3]/2)

    return box_corner

def preprocess_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    # canvas[ (h-new_h)//2:(h-new_h)//2 + new_h , (w-new_w)//2:(w-new_w)//2 + new_w  ,  :] = resized_image
    canvas[0:new_h, 0:new_w,  :] = resized_image

    return canvas


def get_input_to_network(img, inp_dim):
    # img = cv2.imread(img_path)
    img = preprocess_image(img, inp_dim)          #Resize to the input dimension
    # img = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
    img_ =  img[:,:,::-1].transpose((2,0,1)).copy()  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_



def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):

    """
    -box confidence is Pc
    -boxes are the boxes with their corresponding 4 values
    -box_class_probs are the scores or the probabilities of each class in a given bounding box
    -threshold is the value by which to filter.
    """

    box_scores = box_confidence.unsqueeze(1) * box_class_probs

    box_class_scores, box_classes = torch.max(box_scores, dim=-1) # the actual probability of the class

    filtering_mask = box_class_scores >= threshold

    assert box_class_scores.shape == filtering_mask.shape

    scores = torch.masked_select(box_class_scores, filtering_mask)

    boxes = torch.masked_select(boxes, filtering_mask.unsqueeze(1).repeat(1, 4)).view(-1, 4)

    classes = torch.masked_select(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, iou_threshold=.5):

    nms_indices = ops.nms(boxes, scores, iou_threshold)

    scores = torch.index_select(scores, dim=0, index=nms_indices)
    boxes = torch.index_select(boxes, dim=0, index=nms_indices)
    classes = torch.index_select(classes, dim=0, index=nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_output):

    box_xy = yolo_output[:, :2]
    box_wh = yolo_output[:, 2:4]

    box_confidence = yolo_output[ :, 4]
    box_class_probs = yolo_output[:, 5:]

    # boxes = yolo_boxes_to_corners(yolo_output[:, :4])
    boxes = yolo_boxes_to_corners(yolo_output[:, :4])

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    return scores, boxes, classes

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((yi2 - yi1), 0) * max((xi2 - xi1), 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou
