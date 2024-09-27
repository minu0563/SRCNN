import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_paths, input_size, output_size, stride=14, upscale_factor=3):
        super(CustomDataset, self).__init__()
        self.img_paths = glob.glob(img_paths + '/*.png')
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.sub_lr_imgs = []
        self.sub_hr_imgs = []
        self.input_size = input_size
        self.output_size = output_size
        self.pad = abs(self.input_size - self.output_size) // 2

        print("Start {} Images Pre-Processing".format(len(self.img_paths)))
        for img_path in self.img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Unable to load image {img_path}")
                continue

            h = img.shape[0] - np.mod(img.shape[0], self.upscale_factor)
            w = img.shape[1] - np.mod(img.shape[1], self.upscale_factor)
            img = img[:h, :w, :]

            label = img.astype(np.float32) / 255.0
            temp_input = cv2.resize(label, dsize=(0, 0), fx=1/self.upscale_factor, fy=1/self.upscale_factor, interpolation=cv2.INTER_AREA)
            input_img = cv2.resize(temp_input, dsize=(0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)

            for h in range(0, input_img.shape[0] - self.input_size + 1, self.stride):
                for w in range(0, input_img.shape[1] - self.input_size + 1, self.stride):
                    sub_lr_img = input_img[h:h+self.input_size, w:w+self.input_size, :]
                    sub_hr_img = label[h+self.pad:h+self.pad+self.output_size, w+self.pad:w+self.pad+self.output_size, :]

                    sub_lr_img = sub_lr_img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
                    sub_hr_img = sub_hr_img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)

                    self.sub_lr_imgs.append(sub_lr_img)
                    self.sub_hr_imgs.append(sub_hr_img)

        self.sub_lr_imgs = np.asarray(self.sub_lr_imgs)
        self.sub_hr_imgs = np.asarray(self.sub_hr_imgs)
        print("Finish, Created {} Sub-Images".format(len(self.sub_lr_imgs)))

    def __len__(self):
        return len(self.sub_lr_imgs)

    def __getitem__(self, idx):
        lr_img = self.sub_lr_imgs[idx]
        hr_img = self.sub_hr_imgs[idx]
        return lr_img, hr_img
