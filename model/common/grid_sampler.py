# import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def nearest_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, dtype=tf.int32)
    max_x = tf.cast(W - 1, dtype=tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, dtype=tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, dtype=tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i)
    x = tf.cast(tf.round(x), dtype=tf.int32)
    y = tf.cast(tf.round(y), dtype=tf.int32)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x = tf.clip_by_value(x, zero, max_x)
    y = tf.clip_by_value(y, zero, max_y)

    # get pixel value at corner coords
    out = get_pixel_value(img, x, y)

    return out


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


# grid = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing='ij'), axis=-1)
# grid = grid[np.newaxis, :, :]
# grid = tf.convert_to_tensor(grid, dtype=tf.float32)
# grid -= 4.5
# grid /= 4.501
#
# img = np.sum(grid, axis=-1)
# img = np.stack((img, img, img), axis=-1)
# img = tf.convert_to_tensor(img, dtype=tf.float32)
# img += 2
# img *= 50
#
# res = nearest_sampler(img, grid[..., 0], grid[..., 1])
# plt.imshow(np.array(res[0, ...]))
# plt.show()