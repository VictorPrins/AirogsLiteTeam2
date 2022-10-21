import os
import cv2

# most of the code used below was based on the tutorials
# on the opencv website

# directories for running on lisa

# original_images = 'data/test_predictions_3'
# wide_cropped = 'data/zoomed_original'

# dest_dir_224 = 'data/data_augmentation/images_224'
# dest_dir_384 = 'data/data_augmentation/images_384'
# dest_dir_hsv = 'data/data_augmentation/hsv'
# dest_dir_contrast = 'data/data_augmentation/contrast'
# dest_dir_reflection = 'data/data_augmentation/reflection'
# dest_dir_rotation = 'data/data_augmentation/rotation'

# wide_cropped_224 = 'data/data_augmentation/wide_cropped_224'

# paths to run locally
original_images = 'test_predictions_test_data'

dest_dir_224 = 'test_predictions_224'
dest_dir_384 = 'test_predictions_384'
dest_dir_hsv = 'hsv'
dest_dir_contrast = 'contrast'
dest_dir_reflection = 'reflection'
dest_dir_rotation = 'rotation'

test_dir = 'test_'
test_dir_contrast = 'test_contrast'

def resolution(data, dest_dir, dsize):

    for img_name in os.listdir(data):
        if img_name.startswith('zoomed_'):
            next(data)
        else:
            img = cv2.imread(data + '/' + img_name)
            resized = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA )
            cv2.imwrite(dest_dir + '/' + img_name, resized)
    print('done')

def hsv(data, dest_dir):
    for img_name in os.listdir(data):
        
        img = cv2.imread(data + '/' + img_name)

        # equalizes the value channel in the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        equ = cv2.equalizeHist(v)
        merged = cv2.merge([h, s, equ])
        interm_image = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

        # equalized the y channel in the YCrCb color space
        ycrcb = cv2.cvtColor(interm_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        new_y = cv2.equalizeHist(y)
        merged = cv2.merge([new_y, cr, cb])
        final = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # write original and augmented
        cv2.imwrite( dest_dir + '/' + img_name, img)
        cv2.imwrite(dest_dir + '/hsv_' + img_name, final)
    print('done')


# the code below that was used for increasing the contrast of the image was
# done by using the solution provided in a reaction to the stack overflow post below:
# 
# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv

def contrast(data, dest_dir):
    for img_name in os.listdir(data):
        
        img = cv2.imread(data + '/' + img_name)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge([cl,a,b])
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # write augmented and original
        cv2.imwrite( dest_dir + '/' + img_name, img)
        cv2.imwrite(dest_dir + '/contrast_' + img_name, enhanced_img)
    print('done')

def reflection(data, dest_dir):
    for img_name in os.listdir(data):
        
        img = cv2.imread(data + '/' + img_name)

        flip_V = cv2.flip(img, 0)
        flip_H = cv2.flip(img, 1)
        
        cv2.imwrite( dest_dir + '/' + img_name, img)
        cv2.imwrite( dest_dir + '/flippedvertical_' + img_name, flip_V)
        cv2.imwrite( dest_dir + '/flippedhorizontal_' + img_name, flip_H) 
    print('done')

def rotation(data, dest_dir):
    for img_name in os.listdir(data):
        
        img = cv2.imread(data + '/' + img_name)

        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        cv2.imwrite( dest_dir + '/' + img_name, img)
        cv2.imwrite( dest_dir + '/rotated_' + img_name, rotated_img)
    print('done')

dsize = (224, 224)

resolution(original_images, test_dir, dsize)
contrast(test_dir, test_dir_contrast)

print('done all')






        
