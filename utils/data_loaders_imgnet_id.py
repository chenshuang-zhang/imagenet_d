import torch, os
from PIL import Image

import torch
import json
class ImageNetDLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  test_base_dir, few_test=None, transform=None, center_crop=False
                  ):
        super().__init__()

        self.test_path = test_base_dir
        self.categories_list = os.listdir(self.test_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test

        self.transforms=transform

        with open('preprocessing/imgnet_d2imgnet_id.txt') as f:
            self.dict_imgnet_d2imagenet_id = json.load(f)

        for each in self.categories_list:
            folder_path = os.path.join(self.test_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                self.label_lists.append(self.dict_imgnet_d2imagenet_id[each]+[-1]*(10-len(self.dict_imgnet_d2imagenet_id[each]))) 

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
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
    
