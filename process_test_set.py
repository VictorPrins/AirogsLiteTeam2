import torch 
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os
from dataset import AirogsDataset, collate_fn
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv

model = ViTForImageClassification.from_pretrained('./models/best_model')
model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')

all_image_files = os.listdir('data/testset')

with open('submission.csv', 'w') as f:
    f.write('aimi_id; rg_likelihood\n')

    with torch.no_grad():
        for filename in tqdm(all_image_files):

            im = Image.open(os.path.join('data/testset/', filename))
            encoding = feature_extractor(images=im, return_tensors="pt")
            pixel_values = encoding['pixel_values']
            outputs = model(pixel_values)
            logit = outputs.logits

            prob = torch.sigmoid(logit).item()
            aimi_id = filename.split('_')[1][:-4]

            f.write(aimi_id + ';' + str(prob) + '\n')




