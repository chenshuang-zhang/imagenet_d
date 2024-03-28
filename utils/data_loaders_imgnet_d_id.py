import torch, os
from PIL import Image
import torch
import json

class ImageNetDIDLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  test_base_dir, dataset='imagenet_d', transform=None):
        super().__init__()

        self.test_path = test_base_dir
        self.categories_list = os.listdir(self.test_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.dataset = dataset
        self.transforms=transform

        with open('preprocessing/imgnet_d_dir2imgnet_d_id.txt', 'r') as f:
            self.dict_2imgnet_d_id = json.load(f)

        for each in self.categories_list:
            folder_path = os.path.join(self.test_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                self.label_lists.append(self.dict_2imgnet_d_id[each][0]) 

    def __len__(self):
        return len(self.label_lists)

    def _transform(self, sample):
        return self.transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
        img = Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]
        return {"images": img_tensor, "labels": labels, "path": path_list}