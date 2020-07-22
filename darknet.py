# from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import predict_transform, scale_boxes, yolo_boxes_to_corners
import cv2
from torchvision import ops


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.read().split('\n')
        lines = [line for line in lines if len(line) > 0]
        lines = [line for line in lines if line[0] != '#']
        lines = [line.strip() for line in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()

    prev_filters = 3 #keep track of the # of filters in the previous conv layer.
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #Convolutional layer
        if x['type'] == 'convolutional':
            activation = x['activation']

            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            padding = kernel_size // 2
            stride = int(x['stride'])

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
            module.add_module(f'conv_{index}', conv)


            if batch_normalize:
                BN = nn.BatchNorm2d(filters)
                module.add_module(f'BatchNorm2d_{index}', BN)

            if activation == 'leaky':
                module.add_module(f'leaky{index}', nn.LeakyReLU(0.1, inplace=True))

        #TODO: Why are we using Upsampling
        #Upsampling layer
        elif x['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(x['stride']), mode='nearest')
            module.add_module(f'upsample_{index}', upsample)

        elif x['type'] == 'route':
            x['layers'] = [int(layer) for layer in x['layers'].split(',')] # retrieve the route layers

            #here we make the assamption that start is always negative and end is always positive

            #TODO: rename the variables start and end to reflet reality
            #start of a route
            start = x['layers'][0]

            try:
                end = x['layers'][1]
                end = end - index
            except IndexError as e:
                end = 0

            if end < 0:
                filters = output_filters[start + index] + output_filters[end + index]
            else:
                filters = output_filters[start + index]

            route = EmptyLayer()
            module.add_module(f'route_{index}', route)



        elif x['type'] == 'shortcut':

            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)

        elif x['type'] == 'yolo': #this is the detection layer
             mask = [int(x) for x in x['mask'].split(',')]

             anchors = [int(anchor) for anchor in x['anchors'].split(',')]
             anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
             anchors = [anchors[i] for i in mask]

             detection = DetectionLayer(anchors)
             module.add_module(f'detection_{index}', detection)

        else:
            raise ValueError('No such layer: ', x['type'])

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list

#the daknet network
class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()

        self.blocks = parse_cfg(cfg_file) #this is a dict containing all the layers of Darknet loaded from the cfg file

        #module_list is a nn.ModuleList that contains sequential modules each containing exactly one layer of Darknet supplied by self.blocks
        self.net_info, self.module_list = create_modules(self.blocks)


    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} #save feature maps for later use.

        write = False
        for i, module in enumerate(modules):
             module_type = module['type']

             if module_type == 'convolutional' or module_type == 'upsample':
                 x = self.module_list[i](x)

                 # outputs[i] = x

             elif module_type == 'route':

                layers = [int(layer) for layer in module['layers']]


                if len(layers) == 1: #here comes the detection layers and there for to detect in another scale we bring the feature map before the detection layers and feee it to an upsampling layer and continue.
                    assert layers[0]
                    x = outputs[i + layers[0]]

                else: #otherwise here we bring the feature maps of early layers and concatinate it with the current layer in order to benifit from the fine-grained features
                    assert layers[0] < 0 and layers[1] > 0
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[layers[1]]

                    x = torch.cat((map1, map2), dim=1) # the tensor is of shape B, C, W, H, we concatinate along the C (channel wise)

                # outputs[i] = x
             elif module_type == 'shortcut':
                 from_ = int(module['from'])
                 x = outputs[i - 1] + outputs[i + from_]
                # outputs[i] = x

             elif module_type == 'yolo':
                 anchors = self.module_list[i][0].anchors

                 input_dim = int(self.net_info['height'])
                 num_classes = int(module['classes'])

                 x = x.data #get back the actual data of the FloatTensor
                 x = predict_transform(x, input_dim, anchors, num_classes, CUDA)

                 if not write:
                     detection = x
                     write = True

                 else:
                     detection = torch.cat((detection, x), 1)

             outputs[i] = x

        return detection

    def load_weights(self, weights_file):

        """
            only the conv and batch norm layers contain weights
        """

        wf = open(weights_file, 'rb')

        # the first 5 elements are header values
        header = np.fromfile(wf, dtype=np.int32, count=5)

        weights = np.fromfile(wf, dtype=np.float32)

        #TODO: change to more appropriate name
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            if module_type == 'convolutional':
                model = self.module_list[i] # extract the sequential layer.

                #check first if the conv layer contains batch_normalize layers
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0] #extract the conv layer from the sequencial layer.

                if batch_normalize:
                    BN = model[1]

                    #get the number of biases element in the bias tensor of batch norm
                    num_bn_bias = BN.bias.numel()

                    #load the weights
                    bn_bias = torch.from_numpy(weights[ptr:ptr + num_bn_bias])
                    ptr += num_bn_bias

                    # num_bn_weights = bn.weights.numel()
                    assert BN.bias.numel() == BN.weight.numel()


                    bn_weight = torch.from_numpy(weights[ptr:ptr + num_bn_bias])
                    ptr += num_bn_bias

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_bias])
                    ptr += num_bn_bias

                    bn_running_variance = torch.from_numpy(weights[ptr:ptr + num_bn_bias])
                    ptr += num_bn_bias

                    assert BN.bias.data.size() == bn_bias.size()

                    bn_bias = bn_bias.view_as(BN.bias.data)
                    bn_weight = bn_weight.view_as(BN.weight.data)
                    bn_running_variance = bn_running_variance.view_as(BN.running_var)
                    bn_running_mean = bn_running_mean.view_as(BN.running_mean)

                    BN.bias.data.copy_(bn_bias)
                    BN.weight.data.copy_(bn_weight)
                    BN.running_mean.copy_(bn_running_mean)
                    BN.running_var.copy_(bn_running_variance)

                else:
                    num_bias = conv.bias.numel()


                    conv_bias = torch.from_numpy(weights[ptr:ptr + num_bias])
                    ptr += num_bias

                    conv_biases = conv_bias.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_bias)


                num_weights = conv.weight.numel()


                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)






def get_test_input(img):
    # img = cv2.imread(img_path)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
#
model = Darknet("yolov3.cfg")
model.load_weights('yolov3.weights')

inp = cv2.imread('person.jpg')

image_shape = inp.shape[:2]

output = model(get_test_input(inp), torch.cuda.is_available()).squeeze()


# output = output.numpy().squeeze()

print(output.shape)

# assert False
#
LABELS = open('coco.names').read().strip().split("\n")
# # print(LABELS)
# # initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


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

    # scores = box_class_scores * filtering_mask
    # scores = scores[torch.nonzero(scores).squeeze()]

    scores = torch.masked_select(box_class_scores, filtering_mask)

    boxes = torch.masked_select(boxes, filtering_mask.unsqueeze(1).repeat(1, 4)).view(-1, 4)

    # classes = box_classes * filtering_mask
    # classes = classes[torch.nonzero(classes).squeeze()]
    classes = torch.masked_select(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, iou_threshold=.5):

    nms_indices = ops.nms(boxes, scores, iou_threshold)

    scores = torch.gather(scores, dim=0, index=nms_indices)
    boxes = torch.index_select(boxes, dim=0, index=nms_indices)
    classes = torch.gather(classes, dim=0, index=nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_output, images_shape):

    box_xy = yolo_output[:, :2]
    box_wh = yolo_output[:, 2:4]

    box_confidence = yolo_output[ :, 4]
    box_class_probs = yolo_output[:, 5:]

    boxes = yolo_boxes_to_corners(yolo_output[:, :2], yolo_output[:, 2:4])

    # print(f'classes shape before filtering: {boxes.shape}')

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs)

    boxes = scale_boxes(boxes, images_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    return scores, boxes, classes


scores, boxes, classes = yolo_eval(output, images_shape=image_shape)

print(f'scores shape: {scores.shape}')
print(f'boxes shape: {boxes.shape}')
print(f'classes shape: {classes.shape}')

boxes = boxes.numpy()


LABELS = open('coco.names').read().strip().split("\n")
# print(LABELS)
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

for index, i in enumerate(classes.numpy()):
    print(i)
    # extract the bounding box coordinates
    (x, y) = (boxes[index, 0], boxes[index, 1])
    (w, h) = (boxes[index, 2], boxes[index, 3])
    # draw a bounding box rectangle and label on the image
    color = [int(c) for c in COLORS[i]]

    cv2.rectangle(inp, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    text = "{}".format(LABELS[i])
    cv2.putText(inp, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2)






# boxes = []
# confidences = []
# classIDs = []
#
# for detection in output:
#     # extract the class ID and confidence (i.e., probability) of
#     # the current object detection
#     scores = detection[5:]
#     classID = np.argmax(scores)
#     confidence = scores[classID]
#     # filter out weak predictions by ensuring the detected
#     # probability is greater than the minimum probability
#     if confidence > .7:
#         # scale the bounding box coordinates back relative to the
#         # size of the image, keeping in mind that YOLO actually
#         # returns the center (x, y)-coordinates of the bounding
#         # box followed by the boxes' width and height
#         box = detection[0:4]
#         # box = detection[0:4] * np.array([W, H, W, H])
#         (centerX, centerY, width, height) = box.astype("int")
#         # use the center (x, y)-coordinates to derive the top and
#         # and left corner of the bounding box
#         x = int(centerX - (width / 2))
#         y = int(centerY - (height / 2))
#         # update our list of bounding box coordinates, confidences,
#         # and class IDs
#         boxes.append([x, y, int(width), int(height)])
#         confidences.append(float(confidence))
#         classIDs.append(classID)
#
# idxs = cv2.dnn.NMSBoxes(boxes, confidences, .5, .6)
#
# print(idxs)
#
# if len(idxs) > 0:
#     print('detections')
#     # loop over the indexes we are keeping
#     for i in idxs.flatten():
#     # extract the bounding box coordinates
#         (x, y) = (boxes[i][0], boxes[i][1])
#         (w, h) = (boxes[i][2], boxes[i][3])
#
#         color = [int(c) for c in COLORS[classIDs[i]]]
#
#         cv2.rectangle(inp, (x, y), (x + w, y + h), color, 1)
#         # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#         text = "{}".format(LABELS[classIDs[i]])
#         cv2.putText(inp, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#         	0.5, color, 2)
# else:
#     print('no detections')
# # #
# # #
cv2.imshow("Image", inp)
cv2.waitKey(0)
cv2.destroyAllWindows()
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print (pred[0, 0, :])

# print(create_modules(parse_cfg('yolov3.cfg')))
