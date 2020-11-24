import os
import glob
import numpy as np
from PIL import Image
import math
path_image = './data/uavid/val/image'
path_label = './data/uavid/val/label'

save_crop_image = './data/uavid_crop/val/image'
save_crop_label = './data/uavid_crop/val/label'


# size = 6000
crop_size = 512
# batch = size // crop_size

overlap = 0.5
for fp in glob.glob(path_label + '/*.png'):
    PIL_label = Image.open(fp)
    print(fp)
    # assert PIL_label.size == (size, size), 'Label {} information wrong!'.format(fp)
    (height, width) = PIL_label.size
    step = math.floor(crop_size*(1-overlap))
    batch_h = (height-crop_size) // (step) +1
    batch_w = (width-crop_size) // (step) +1
    for h in range(batch_h + 1):
        for w in range(batch_w + 1):
            label_basename = os.path.basename(fp).split('_noBoundary.')[0]
            label_save_name = label_basename + '_1.0_' + str(h) + '_' + str(w) + '.png'
            label_save_path = os.path.join(save_crop_label, label_save_name)

            if h == batch_h and w != batch_w:
                crop_label = PIL_label.crop([(height - crop_size), (w*step), height, (crop_size + w*step)])
            elif h != batch_h and w == batch_w:
                crop_label = PIL_label.crop([(h*step), (width - crop_size), (crop_size + h*step), width])
            elif h == batch_h and w == batch_w:
                crop_label = PIL_label.crop([(height - crop_size), (width - crop_size), height, width])
            else:
                crop_label = PIL_label.crop([(h*step), (w*step),
                                             (crop_size+h*step), (crop_size+w*step)])

            crop_label.save(label_save_path)
            


for fp in glob.glob(path_image + '/*.png'):
    PIL_image = Image.open(fp)
    print(fp)
    # assert PIL_image.size == (size, size), 'Image {} information wrong!'.format(fp)
    (height, width) = PIL_image.size
    step = math.floor(crop_size*(1-overlap))
    batch_h = (height-crop_size) // (step) +1
    batch_w = (width-crop_size) // (step) +1
    for h in range(batch_h + 1):
        for w in range(batch_w + 1):
            image_basename = os.path.basename(fp).split('.')[0]
            image_save_name = image_basename + '_1.0_' + str(h) + '_' + str(w) + '.png'
            image_save_path = os.path.join(save_crop_image, image_save_name)

            if h == batch_h and w != batch_w:
                crop_image = PIL_image.crop([(height - crop_size), (w*step), height, (crop_size + w*step)])
            elif h != batch_h and w == batch_w:
                crop_image = PIL_image.crop([(h*step), (width - crop_size), (crop_size + h*step), width])
            elif h == batch_h and w == batch_w:
                crop_image = PIL_image.crop([(height - crop_size), (width - crop_size), height, width])
            else:
                crop_image = PIL_image.crop([(h*step), (w*step),
                                             (crop_size+h*step), (crop_size+w*step)])
            crop_image.save(image_save_path)



