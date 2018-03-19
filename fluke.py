import numpy as np
import skimage
from skimage import io, transform, segmentation, morphology, color
import pathlib
import matplotlib.pyplot as plt



def raw_moment(i,j, mask):
    height, width = mask.shape

    def foo(width, power):
        #creates a vector to be used to create meshgrid
        x = list(range(width))
        x = np.subtract(x, width/2) #subtract mean (centers picture)
        return np.power(x, power)

    x = foo(width, i)
    y = foo(height, j)

    x, y = np.meshgrid(x, y)

    xy = np.multiply(x, y)

    mask = np.multiply(mask, xy)

    return np.sum(mask)

def gen_features(mask):
    # returns a vector a features include all moments up to second moment

    f = []
    for i in range(3):
        for j in range(3):
            f.append(raw_moment(i,j,mask))

    return f

def predict_tail(labels):
    # assume the background is primarily labeled on the border, making the tail the other label
    temp = []
    temp.extend(labels[0, :])  # top row
    temp.extend(labels[-1, :])  # bottom row
    temp.extend(labels[:, 0])  # first col
    temp.extend(labels[:, -1])  # last col

    ones = np.count_nonzero(temp)
    zeros = len(temp) - ones


    if ones < zeros:
        tail = np.array(labels) # tail is already represented by the ones class
    else:
        tail = np.abs(np.subtract(1, np.array(labels))) # tail is now represented by the ones class

    return skimage.img_as_ubyte(tail)

def predict_tail2(labels):
    # more sophisticated
    global pipe



def process_img(img):


    s = img.shape

    # downsample
    downscale = s[1] // 100
    if downscale > 1:
        thumb = transform.pyramid_reduce(img, downscale=downscale)
        thumb = 255 * thumb
        thumb = thumb.astype('uint8')
    else:
        thumb = img.astype('uint8')

    # slic
    labels = segmentation.slic(thumb, compactness=.1, n_segments=2)

    # predict tail segment
    tail_mask = predict_tail2(labels)

    # upsample mask
    tail_mask_big = transform.resize(255*tail_mask, output_shape=(s[0], s[1]), order=0, mode='edge')#refine sizing
    tail_mask_big = skimage.img_as_ubyte(tail_mask_big)

    # dialate tail mask
    tail_mask_big = morphology.binary_dilation(tail_mask_big, selem=morphology.disk(radius=downscale))

    # generate full res tail
    alpha = np.expand_dims(255*tail_mask_big, axis=2)
    img = np.concatenate((img, alpha), axis=2)
    img = skimage.img_as_ubyte(img)

    # returns thumbnail, and full resolution img with alpha channels
    return img

def im_loader(file, img_num):
    img = io.imread(file)
    s = img.shape

    if len(s) == 2: #greyscale
        img = color.gray2rgb(img)
    elif len(s) == 3:
        if s[2] == 4: #remove alpha channel (this allows me to continually refine runs)
            img = img[:,:,0:3]

    return img


def process_dir(input_path, output_path):
    # creates an image collection from input_path and saves results to output_path
    ic = io.ImageCollection(input_path, load_func=im_loader)

    #then process each img
    for i in range(len(ic)):
        img = process_img(ic[i])
        fname = pathlib.Path(ic.files[i])
        print('Saving {}.png'.format(fname.stem))
        io.imsave(output_path.joinpath(fname.stem + '.png'), img)

def main():
    input_pattern = 'C:/Users/felpsdl/PycharmProjects/happywhale/working/*.png'
    output_path = pathlib.Path('C:/Users/felpsdl/PycharmProjects/happywhale/working2/')
    process_dir(input_pattern, output_path)

def analyze_tail_moment(folder):
    from sklearn import preprocessing, pipeline, svm, metrics, linear_model

    #assume this folder contains human vetted images
    #will use the alpha channel to calculate the moment for tail/background segments

    # X_train, X_test, y_train, y_test = build_moment_dataset(folder)
    # np.savez('tail_xy.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    tail_xy = np.load('tail_xy.npz')
    X_train = tail_xy['X_train']
    X_test = tail_xy['X_test']
    y_train = tail_xy['y_train']
    y_test =  tail_xy['y_test']

    # train a classifier
    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("Prediction accuracy - {}".format(metrics.accuracy_score(y_test, y_pred)))
    return pipe

'''GLOBAL VARIABLE WARNING'''
pipe = analyze_tail_moment('')

def build_moment_dataset(folder):

    from sklearn import model_selection

    def just_alpha(file, img_num):
        img = io.imread(file)
        img = np.squeeze(img[:,:,3])
        img = np.bool_(img)
        return img

    ic = io.ImageCollection(folder, load_func=just_alpha)

    X_pos = np.zeros(shape=(len(ic), 9), dtype='float')
    y_pos = np.ones(shape=(len(ic)), dtype='int')
    X_neg = np.zeros_like(X_pos)
    y_neg = np.zeros_like(y_pos)

    for i in range(len(ic)):
        #calculate the positive and negative moments
        alpha = ic[i]
        X_pos[i,:] = gen_features(alpha)
        X_neg[i, :] = gen_features(np.logical_not(alpha))

    X = np.concatenate((X_neg, X_pos))
    y = np.concatenate((y_neg, y_pos))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=.5, shuffle=True, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    folder = 'C:/Users/felpsdl/PycharmProjects/happywhale/final/*.png'
    analyze_tail_moment(folder)

