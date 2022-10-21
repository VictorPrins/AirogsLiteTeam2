import torch 
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os
from dataset import AirogsDataset, collate_fn
import json
import numpy as np
import matplotlib.pyplot as plt

# im = Image.open('data/images/DEV00071.jpg')

dataset = AirogsDataset()

with open('misc/output_full_run.out') as file:
    lines = file.readlines()

lines = [line.rstrip() for line in lines]

image_files_list = lines[0]
index = image_files_list.index('[')
image_files_list = image_files_list[index:].replace("'", '"')
image_files_list = json.loads(image_files_list)

# overwrite the image_files list with the order from Lisa, so we can index with the indices from Lisa
dataset.image_files = image_files_list

# get the validation indices from the output file
val_set_indices = lines[1]
index = val_set_indices.index('[')
val_set_indices = val_set_indices[index:]
val_set_indices = json.loads(val_set_indices)

val_set = torch.utils.data.Subset(dataset, val_set_indices)

dataloader = torch.utils.data.DataLoader(val_set, batch_size=8, collate_fn=collate_fn)

# This is the model corresponding to output_full_run.out, or equivalently run 10168093
model = ViTForImageClassification.from_pretrained('./models/checkpoint-7350')

def compute_loss(model, inputs, return_outputs=False):

    # labels is size [batch_size]
    labels = inputs.get('labels').to(torch.float)

    # forward pass
    outputs = model(**inputs)

    # get logits and squeeze into size [batch_size]
    logits = outputs.get('logits').squeeze(dim=1)

    
    imbalance_weights = torch.zeros(labels.size()).to(self.device)

    for idx, label in enumerate(labels):
        imbalance_weights[idx] = self.class_weights[int(label)]

    loss_fn = torch.nn.BCEWithLogitsLoss(weight=imbalance_weights)

    loss = loss_fn(logits, labels)

    return (loss, outputs) if return_outputs else loss

eval_losses=[]
eval_accu=[]


model.eval()

losses = []

i = 0

with torch.no_grad():
    for inputs in dataloader:
   
        labels = inputs.get('labels').to(torch.float)

        outputs = model(**inputs)

        logits = outputs.get('logits').squeeze(dim=1)


        imbalance_weights = torch.zeros(labels.size())

        class_weights = torch.tensor([1,9])
        for idx, label in enumerate(labels):
            imbalance_weights[idx] = class_weights[int(label)]

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=imbalance_weights, reduction='none')

        loss = loss_fn(logits, labels)

        losses.extend(loss.numpy())

        i+=1
        print(i)
        # print(torch.sigmoid(logits))
        # print(labels)
        if i == 200:
            break


print(losses)
print(np.mean(losses))




# feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
# encoding = feature_extractor(images=im, return_tensors="pt")
# pixel_values = encoding['pixel_values']




# outputs = model(pixel_values)
# logit = outputs.logits
# prob = torch.nn.functional.sigmoid(logit)

# print("Predicted class:", 'RG' if prob > 0.5 else 'NRG')