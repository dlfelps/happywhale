from skimage import segmentation, future, io, color, morphology
import pathlib
import numpy as np
from PIL import Image

def extract_background(rgba):
    # rgba: an image with dimensions [width, height, 4 (rgba)]
    # assumes current background is set to 0 on the alpha channel
    # removes foreground and sets equal to average of background values
    pass

if __name__ == "__main__":

    file = pathlib.Path('C:/Users/felpsdl/PycharmProjects/happywhale/tail.jpg')
    img = io.imread(file)
    s = img.shape

    mask = future.manual_lasso_segmentation(img)
    # io.imshow(mask)
    # io.show()

    '''
    labels = segmentation.slic(img, n_segments=5000, compactness=10.0, sigma=0)
    # out = color.label2rgb(labels, img, kind='avg')
    # io.imshow(out)
    # io.show()

    # filter any clusters that are not total contained within the mask
    all = np.unique(labels)
    background = np.multiply(labels, np.logical_not(mask))
    background = np.unique(background) #these are the clusters in the background
    tail_set = np.setdiff1d(all, background)

    # refine the mask
    rf = np.zeros_like(mask, dtype='bool')
    for x in range(s[0]):
        for y in range(s[1]):
            if labels[x,y] in tail_set:
                rf[x,y] = True
    '''

    # morphological operations
    rf = morphology.binary_dilation(mask, morphology.disk(radius=10))

    # io.imshow(rf*img)
    # io.show()

    # write tail to file
    rgba = np.zeros(shape=(s[0], s[1], 4), dtype='uint8')
    rgba[:,:,0:3]=img
    rgba[:,:,3]=rf*255
    np.save('tail.npy', rgba)
    io.imshow(rgba)
    io.show()





