# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# img = Image.open('horses.jpg').convert('RGB')
#
# img = np.array(img).transpose((2, 0, 1))
#
# print(img.shape)
#
# img = np.expand_dims(img, axis=0)
#
# print(img.shape)
#
#
#
# img = nn.Upsample(scale_factor=2, mode='nearest')(torch.from_numpy(img).type(torch.FloatTensor))
#
# img = img.numpy().squeeze().transpose((1, 2, 0))
#
# print(img.shape)
#
# plt.imshow(img)
# plt.show()

def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer


        write = 0
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":

                x = self.module_list[i](x)
                outputs[i] = x


            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]


                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x



            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])

                #Get the number of classes
                num_classes = int (modules[i]["classes"])

                #Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue


                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]



        try:
            return detections
        except:
            return 0
