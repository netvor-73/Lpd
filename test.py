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
import torch
import numpy as np


# a = torch.randn(10,)
#
# b = torch.tensor([False, False, False, False, False, False, False, False, True, True])
#
# # x = a * b.unsqueeze(1)
# print(a)
# print(torch.masked_select(a, b))


anchors = np.array([[10,13], [16,30], [33,23]])
nm_anchors = len(anchors)

anchors_tensor = torch.from_numpy(anchors).view(1, 1, 1, nm_anchors, 2)

print(anchors_tensor)

a = torch.randn(1, 3, 13, 13)

conv_dims = a.shape[1:3]

anchors_tensor = anchors_tensor.repeat(13 * 13, 1).unsqueeze(0)

print(anchors_tensor)
