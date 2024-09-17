import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from skimage import io
from tqdm.notebook import tqdm
from keras import layers
from keras import models
from multiprocessing import Pool
from functools import partial
from scipy import ndimage

# -----------------------------PREPROCESSING-----------------------------------------------
def random_slice(shape, target_size=(256,256)):
    height, width=shape
    crop_x=np.random.randint(0, width-target_size[0]+1)
    crop_y=np.random.randint(0, height-target_size[1]+1)
    return slice(crop_y, crop_y+target_size[1]), slice(crop_x, crop_x+target_size[0])

def slice_data(img, window_x, window_y):
    n_blocks=np.array(img.shape)//[window_y, window_x]
    big_window_y, big_window_x=(img.shape/n_blocks).astype('int')
    window=min(big_window_y, big_window_x) # this is stupid but I don't want to think about where it's window x and where it's window y below. So square windows only

    img = img[:n_blocks[0]*window, :n_blocks[1]*window]

    output = img.reshape((-1, window, img.shape[1]//window, window))
    return np.array(output.transpose((0,2,1,3)).reshape(-1, window, window)[:, :window_x, :window_y])

def random_flip_rotate(X, y, flip=True, rotate=True):
    if flip:
        flip_img=np.random.randint(0,3,size=len(X)) # 0=no flip, 1=flip vertical, 2=flip horizontal
    else:
        flip_img=np.zeros(len(X), dtype='int')
    if rotate:
        rot_img=np.random.randint(0,4, size=len(X)) # n*90 degrees rotation ccw
    else:
        rot_img=np.zeros(len(X), dtype='int')

    transformed_X=flip_rotate(X, flip_img, rot_img)
    transformed_y=flip_rotate(y, flip_img, rot_img)

    return transformed_X, transformed_y

def flip_rotate(img_arr, flip_img, rot_img):
    #flip
    img_arr[flip_img==1]=np.flip(img_arr[flip_img==1], axis=1) # vertical
    img_arr[flip_img==2]=np.flip(img_arr[flip_img==2], axis=2) # horizontal

    #rotate
    for k in np.unique(rot_img):
        img_arr[rot_img==k]=np.rot90(img_arr[rot_img==k], k=k, axes=(1,2))

    return img_arr

def gaussian_parallel(imgs, n_processes=8, show_tqdm=False, **kwargs):
    p=Pool(processes=n_processes)
    if show_tqdm:
        progress_bar=tqdm
        progress_kwargs={'total':len(imgs), 'desc':'computing gaussians'}
    else:
        progress_bar=lambda x: x
        progress_kwargs={}
    out=[x for x in progress_bar(p.imap(partial(ndimage.gaussian_filter,  **kwargs), imgs.astype('float'), chunksize=8), **progress_kwargs)]

    return np.array(out)

def load_data(directory):
    X=io.imread(directory+'/X.tif')
    y=io.imread(directory+'/y.tif')

    geminin_X, pip_X=X[:,0], X[:,1]
    geminin_y, pip_y=y[:,0], y[:,1]
    
    return geminin_X, pip_X, geminin_y, pip_y

def preprocess_data(directory, subtract_background=False, norm_quantile=0.99):
    geminin_X, pip_X, geminin_y, pip_y = load_data(directory)
    
    geminin_X=geminin_X/np.quantile(geminin_X, norm_quantile)
    pip_X=pip_X/np.quantile(pip_X, norm_quantile)
    geminin_y=geminin_y/np.quantile(geminin_y, norm_quantile)
    pip_y=pip_y/np.quantile(pip_y, norm_quantile)
    
    img_size=geminin_X[0].shape
    # interleave images from geminin and pip channels
    X=np.stack([pip_X, geminin_X], axis=1).reshape(-1, *img_size)
    y=np.stack([pip_y, geminin_y], axis=1).reshape(-1, *img_size)

    if subtract_background:
        blur_bg=gaussian_parallel(y, sigma=100, truncate=3, show_tqdm=True)
        y=y-blur_bg
        y-=y.min()
        y=np.clip(y, a_min=0, a_max=65535).astype('uint16')

    return X, y

def train_test_split(X, y, split=0.8):
    cutoff=int(len(X)*split)
    train_X=X[:cutoff]
    train_y=y[:cutoff]
    test_X=X[cutoff:]
    test_y=y[cutoff:]
    return train_X, train_y, test_X, test_y

def get_data(directory, test_split=0.8, flip=True, rotate=True, windows=True, window_size=(256,256), crop_mode='slice', subtract_background=False, norm_quantile=0.99):
    np.random.seed(42)
    X, y = preprocess_data(directory, subtract_background=subtract_background, norm_quantile=norm_quantile)

    dataset_X=[]
    dataset_y=[]
    if windows:
        if crop_mode=='random':
            for X_img, y_img in zip(X, y):
                crop_y, crop_x = random_slice(X_img.shape, window_size)
                dataset_X.append(X_img[crop_y, crop_x])
                dataset_y.append(y_img[crop_y, crop_x])
                
            dataset_X=np.array(dataset_X)
            dataset_y=np.array(dataset_y)
        
        elif crop_mode=='slice':
            for X_img, y_img in zip(X, y):
                X_blocks=slice_data(X_img, *window_size)
                y_blocks=slice_data(y_img, *window_size)
                dataset_X.append(X_blocks)
                dataset_y.append(y_blocks)
            dataset_X=np.concatenate(dataset_X)
            dataset_y=np.concatenate(dataset_y)
        
        elif crop_mode=='tile':
            for X_img, y_img in zip(X, y):
                X_tiles=get_tiles(X_img, window_size[0])
                y_tiles=get_tiles(y_img, window_size[0])
                dataset_X.append(X_tiles)
                dataset_y.append(y_tiles)
            dataset_X=np.concatenate(dataset_X)
            dataset_y=np.concatenate(dataset_y)
            
        else:
            raise ValueError('crop_mode must be one of "random", "slice", or "tile"')

    dataset_X, dataset_y=random_flip_rotate(dataset_X, dataset_y, flip=flip, rotate=rotate)
    
    dataset_X=dataset_X[...,np.newaxis]
    dataset_y=dataset_y[...,np.newaxis]

    return train_test_split(dataset_X, dataset_y, test_split)

def plot_random(dataset_X, dataset_y, rows, cols, figsize=2):
    '''for comparing a random selection of images side by side'''
    n_plots=rows*cols
    n_imgs=len(dataset_X)

    to_plot=np.random.choice(np.arange(n_imgs), size=n_plots, replace=False)
    fig, axes = plt.subplots(rows, 2*cols, figsize=(2*figsize*cols, figsize*rows))
    X=dataset_X[to_plot]
    y=dataset_y[to_plot]
    vmin, vmax=(np.min([X,y]), np.max([X,y]))
    for pair, X_img, y_img in zip(axes.reshape(-1,2), X, y):
        cbar=pair[0].imshow(X_img, vmin=vmin, vmax=vmax)
        plt.colorbar(cbar, ax=pair[0])
        cbar=pair[1].imshow(y_img, vmin=vmin, vmax=vmax)
        plt.colorbar(cbar, ax=pair[1])
    fig.tight_layout()

# break images into 512x512 overlapping patches
def pad_to_shape(img, shape):
    # pad bottom and right to the desired shape
    y_pad=shape[0]-img.shape[0]
    x_pad=shape[1]-img.shape[1]
    return np.pad(img, ((0,y_pad),(0,x_pad)), mode='edge')

def get_tiles(img, tile_size=512):
    y_blocks, x_blocks=np.array(img.shape[-2:])//tile_size+1

    # compute overlap size
    overlap_y, overlap_x=(tile_size-(np.array(img.shape[-2:])%tile_size))//[y_blocks-1, x_blocks-1]
    # get tiles
    tiles=[]
    for y in range(y_blocks):
        for x in range(x_blocks):
            y_start=y*(tile_size-overlap_y)
            y_end=y_start+tile_size
            x_start=x*(tile_size-overlap_x)
            x_end=x_start+tile_size
            tiles.append(pad_to_shape(img[y_start:y_end, x_start:x_end], (tile_size, tile_size)))
    return np.array(tiles)

def stitch_tiles(tiles, output_shape):
    # get number of rows and columns
    tile_size=tiles.shape[-1]
    rows, cols=np.array(output_shape)//tile_size+1

    # initialize output array
    output=np.zeros((rows*tile_size, cols*tile_size))

    # compute overlap size
    overlap_y, overlap_x=(tile_size-(np.array(output_shape)%tile_size))//[rows-1, cols-1]

    # stitch tiles with feathered averaging of overlaps
    for y in range(rows):
        for x in range(cols):
            y_start=y*(tile_size-overlap_y)
            y_end=y_start+tile_size
            x_start=x*(tile_size-overlap_x)
            x_end=x_start+tile_size
            output[y_start:y_end, x_start:x_end]+=tiles[y*cols+x]
    
    for x in range(1,cols):
        x_start=x*(tile_size-overlap_x)
        x_end=x_start+overlap_x
        output[:,x_start:x_end]/=2

    for y in range(1,rows):
        y_start=y*(tile_size-overlap_y)
        y_end=y_start+overlap_y
        output[y_start:y_end,:]/=2
    
    return output[0:output_shape[0], 0:output_shape[1]]

# ----------------------------------WRITE TFRECORDS---------------------------------------
# TFRecords
def write_tfrecord(X, y, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for X_img, y_img in zip(tqdm(X, leave=False, desc=filename), y):
            writer.write(tf.io.serialize_tensor([X_img, y_img]).numpy())

def parse(serialized):
    return tf.io.parse_tensor(serialized, out_type=tf.float64)

def train_X_y(dataset_line):
    parsed_line=parse(dataset_line)
    return parsed_line[0], parsed_line[1]

def read_tfrecord(filepaths, n_parallel=5, shuffle=False, shuffle_buffer=4000, batch_size=24):
    data=tf.data.TFRecordDataset(filepaths, n_parallel_reads=n_parallel)
    if shuffle:
        data=data.shuffle(shuffle_buffer)
    data=data.map(train_X_y, num_parallel_calls=n_parallel)
    return data.batch(batch_size).prefetch(1)

# --------------------------------UNET ARCHITECTURE------------------------------
def Conv2DLayer(x, filters, kernel, strides, padding, block_id, dropout_rate=0.4, kernel_init=tf.keras.initializers.orthogonal(seed=42)):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                      kernel_initializer=kernel_init, name=prefix+'conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(dropout_rate, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def Transpose_Conv2D(x, filters, kernel, strides, padding, block_id, dropout_rate=0.4, kernel_init=tf.keras.initializers.orthogonal(seed=43)):
    prefix = f'block_{block_id}_'
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                               kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(dropout_rate, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def UNet(input_shape, filters=16, dropout_rate=0.4):
    inputs = layers.Input(shape=input_shape)
    # 256->128
    conv0=Conv2DLayer(inputs, filters, 3, strides=1, padding='same', block_id=0, dropout_rate=dropout_rate)
    conv1 = Conv2DLayer(conv0, 2*filters, 5, strides=2, padding='same', block_id=1, dropout_rate=dropout_rate)
    # 128->64
    conv2 = Conv2DLayer(conv1, 4*filters, 5, strides=2, padding='same', block_id=2, dropout_rate=dropout_rate)
    # 64->32
    conv3 = Conv2DLayer(conv2, 4*filters, 3, strides=1, padding='same', block_id=3, dropout_rate=dropout_rate)
    conv4 = Conv2DLayer(conv3, 8*filters, 3, strides=2, padding='same', block_id=4, dropout_rate=dropout_rate)

    # 32->64
    deconv1 = Transpose_Conv2D(conv4, 8*filters, 3, strides=2, padding='same', block_id=5, dropout_rate=dropout_rate)

    # 64->128
    skip = layers.concatenate([deconv1, conv2])
    conv4 = Conv2DLayer(skip, 8*filters, 3, strides=1, padding='same', block_id=6, dropout_rate=dropout_rate)
    deconv2=Transpose_Conv2D(conv4, 4*filters, 5, strides=2, padding='same', block_id=7, dropout_rate=dropout_rate)

    # 128-256
    skip = layers.concatenate([deconv2, conv1])
    deconv3=Transpose_Conv2D(skip, 2*filters, 5, strides=2, padding='same', block_id=8, dropout_rate=dropout_rate)

    skip = layers.concatenate([deconv3, conv0])
    convFinal = layers.Conv2D(1, 3, strides=1, padding='same', activation='relu',
                       kernel_initializer=tf.keras.initializers.orthogonal(seed=44), name='final_conv')(skip)
    
    return models.Model(inputs=inputs, outputs=convFinal)