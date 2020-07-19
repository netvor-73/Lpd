# from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
            module.add_module(f'conv_{index}', conv)


            if batch_normalize:
                BN = nn.BatchNorm2d(filters)

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
