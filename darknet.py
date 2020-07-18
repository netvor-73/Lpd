# from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
            upsample = nn.UpSampling2D(scale_factor=int(x['stride']), mode='nearest')
            module.add_module(f'upsample_{index}', upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',') # retrieve the route layers

            #start of a route
            
