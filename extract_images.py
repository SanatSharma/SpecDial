from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

# Download annotations before doing the below step from http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# Also download dataset

# Set datadirectory to image location
dataDir='..'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


def extract_images(imgIds):

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    # Retrieve images
    imgs = coco.loadImgs(imgIds)

    for img in imgs:
        I = io.imread(img)
        plt.axis('off')
        plt.imshow(I)
        plt.show()

if __name__ == "__main__":

    #Initialize Coco object    
    coco = Coco(annFile)

    imgIds = [127947, 127947, 233663, 532136, 258922, 100334, 153403, 519161, 238379, 62364]

    extract_images(imgIds)