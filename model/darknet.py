# from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from .utils import predict_transform


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

    def load_weights(self, weightfile):

        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)


                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
