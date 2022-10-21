import torch 
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import cv2
import numpy as np

im = Image.open('data/images/DEV01673.jpg')

model = ViTForImageClassification.from_pretrained('./models/checkpoint-7350')

from vit_grad_rollout import VITAttentionGradRollout

grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
encoding = feature_extractor(images=im, return_tensors="pt")
pixel_values = encoding['pixel_values']

# outputs = model(pixel_values)
# logit = outputs.logits
# prob = torch.nn.functional.sigmoid(logit)

# print("Predicted class:", 'RG' if prob > 0.5 else 'NRG')

mask = grad_rollout(pixel_values, category_index=0)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

np_img = np.array(im)[:, :, ::-1]
cv2.imwrite("orig.png", np_img)

mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
mask = show_mask_on_image(np_img, mask)

sidebyside = np.concatenate((np_img, mask), axis=1)
cv2.imshow("Input Imagxe", sidebyside)
cv2.imwrite('sidebyside.png', sidebyside)

cv2.imwrite('mask.png', mask)
cv2.waitKey(-1)