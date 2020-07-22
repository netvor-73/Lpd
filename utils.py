from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, input_dim, anchors, num_classes=10, CUDA=False):

    """
        - prediction: tensor of (B, C, W, H)
        - input_dim: the dimensions of the input image
    """

    batch_size = prediction.size(0)

    stride =  input_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    #assuming we have 10 classes and the W, H = (13, 13)
    # reshape the prediction tensor to (B, 45, W*H)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    #here the anchors have to be a fraction of the the input dimension
    # anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction[:, :, :2] = torch.sigmoid(prediction[:, :, :2]) # center x, y value
    # prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1]) # center y value
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4]) # objectness score

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, x_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # print(prediction[..., ])

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    #apply sigmoid to the class scores
    prediction[:, :, 5:] = torch.sigmoid(prediction[:, :, 5:])
    prediction[:, :, :4] /= stride

    print(prediction[..., :4])


    return prediction

def yolo_boxes_to_corners(box_xy, box_wh):

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return torch.cat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], dim=1)

    # box_corner = boxes.new(boxes.shape)
    # box_corner[:,0] = (boxes[:,0] - boxes[:,2]/2)
    # box_corner[:,1] = (boxes[:,1] - boxes[:,3]/2)
    # box_corner[:,2] = (boxes[:,0] + boxes[:,2]/2)
    # box_corner[:,3] = (boxes[:,1] + boxes[:,3]/2)
    #
    # return box_corner



def scale_boxes(boxes, image_shape):

    height = image_shape[0]
    width = image_shape[1]
    image_dims = torch.from_numpy(np.array([height, width, height, width]))
    image_dims = image_dims.view(1, 4)


    boxes = boxes * image_dims
    return boxes

def write_results(prediction, confidence, num_classes, nms_threshold=.6):
    conf_mask = (prediction[:,:, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for index in range(batch_size):
        image_pred = prediction[index]

        """
        - max_conf: the actual class max scores
        - max_conf_score: the indices of the max scores
        """
        max_conf, max_conf_score = torch.max(image_pred[:, 5:], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
