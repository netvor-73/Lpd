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
import cv2

# a = torch.randn(10,)
#
# b = torch.tensor([False, False, False, False, False, False, False, False, True, True])
#
# # x = a * b.unsqueeze(1)
# print(a)
# print(torch.masked_select(a, b))


# anchors = np.array([[10,13], [16,30], [33,23]])
# nm_anchors = len(anchors)
#
# anchors_tensor = torch.from_numpy(anchors).view(1, 1, 1, nm_anchors, 2)
#
# print(anchors_tensor)
#
# a = torch.randn(1, 3, 13, 13)
#
# conv_dims = a.shape[1:3]
#
# anchors_tensor = anchors_tensor.repeat(13 * 13, 1).unsqueeze(0)
#
# print(anchors_tensor)
img = cv2.imread('images/person.jpg')

img_h, img_w = img.shape[0], img.shape[1]
w, h = (416, 416)
new_w = int(img_w * min(w/img_w, h/img_h))
new_h = int(img_h * min(w/img_w, h/img_h))
resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)



canvas = np.full((416, 416, 3), 255, dtype=np.uint8)


canvas[0:new_h, 0:new_w,  :] = resized_image

canvas = canvas / 255

print(canvas.shape)

cv2.imshow('winname', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

#extremely important comment---------------------------------

# boxes[:, 0] -= (416 - scaling_factor*image_shape[1])/2
# boxes[:, 1] -= (416 - scaling_factor*image_shape[0])/2
# boxes[:, 2] -= (416 - scaling_factor*image_shape[1])/2
# boxes[:, 3] -= (416 - scaling_factor*image_shape[0])/2
