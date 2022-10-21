from cProfile import label
import torch
import os
from PIL import Image
from transformers import ViTFeatureExtractor
import csv

class AirogsDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.image_files = os.listdir('data/images')[:200]
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')

        self.labels = {}
        self.label2id = {'NRG': 0, 'RG': 1}
        self.id2label = {0: 'NRG', 1: 'RG'}

        # first index is the number of samples of class 0, and second index number of samples of class 1 
        self.samples_per_class = [0, 0]

        with open('data/dev_labels.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:

                label_id = self.label2id[row['class']]
                self.samples_per_class[label_id] += 1
                self.labels[row['aimi_id']] = label_id




    def __getitem__(self, idx):
        im_name = self.image_files[idx]
        im = Image.open(os.path.join('data/images/', im_name))

        # preprocess (mainly: resize image to input size for model)
        encoding = self.feature_extractor(images=im, return_tensors="pt")

        return {
            # get the pixel_values and strip the batch_dimension
            'pixel_values': encoding['pixel_values'][0],
            # get the class from self.labels, by stripping the .jpg extension
            'label': self.labels[os.path.splitext(im_name)[0]]
        }

    def __len__(self):
        return len(self.image_files)




def collate_fn(batch):
    # batch is a list of items of the type that AirogsDataset.__getitem__() returns
    # this function must return a dict with keys pixel_values and labels, containing
    # tensors with the respective data with the batch as first dimension

    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }
    