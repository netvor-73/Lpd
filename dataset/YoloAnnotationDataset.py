import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2


class YoloAnnotationDataset(Dataset):

    def __init__(self, root_dir, seed):
        self.root_dir = root_dir
        self.image_names = [image for image in os.listdir(self.root_dir) if '.jpg' in image]
        self.seed = seed

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.tolist()

        np.random.seed(self.seed)
        np.random.shuffle(self.image_names)

        image_path = os.path.join(self.root_dir, self.image_names[index])

        image = cv2.imread(image_path)
        # image = image[:, :, ::-1]


        with open(image_path.replace('.jpg', '.txt')) as f:
            file = f.read().split('\n')
            file = [line for line in file if len(line) > 0]
            file = [i.strip() for i in file]
            bbox = np.array([box.split(' ') for box in file])

        sample = {'image': image, 'bbox': bbox.astype(np.float32).reshape(-1, 5)}

        return sample
