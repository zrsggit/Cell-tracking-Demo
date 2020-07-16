import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.cm as cm
__author__ = 'assafarbelle'


def put_kernels_on_grid (kernel, grid_Y_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    grid_Y, grid_X = grid_Y_X
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    Z = kernel.get_shape()[2]
    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, Z]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, Z]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.summary.image order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8

def my_clustering_loss(net_out,feature_map):

    net_out_vec = tf.reshape(net_out,[-1,1])
    pix_num = net_out_vec.get_shape().as_list()[0]
    feature_vec = tf.reshape(feature_map,[pix_num,-1])
    net_out_vec = tf.div(net_out_vec, tf.reduce_sum(net_out_vec,keep_dims=True))
    not_net_out_vec = tf.subtract(tf.constant(1.),net_out_vec)
    mean_fg_var = tf.get_variable('mean_bg',shape = [feature_vec.get_shape().as_list()[1],1], trainable=False)
    mean_bg_var = tf.get_variable('mean_fg',shape = [feature_vec.get_shape().as_list()[1],1], trainable=False)

    mean_bg = tf.matmul(not_net_out_vec,feature_vec,True)
    mean_fg = tf.matmul(net_out_vec,feature_vec,True)

    feature_square = tf.square(feature_vec)

    loss = tf.add(tf.matmul(net_out_vec, tf.reduce_sum(tf.square(tf.subtract(feature_vec, mean_fg_var)), 1, True), True),
                  tf.matmul(not_net_out_vec, tf.reduce_sum(tf.square(tf.subtract(feature_vec,mean_bg_var)), 1, True), True))
    with tf.control_dependencies([loss]):
        update_mean = tf.group(tf.assign(mean_fg_var,mean_fg),tf.assign(mean_bg_var,mean_bg))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)

    return loss

def plot_segmentation(I,GT,Seg, fig=None):

    I = np.squeeze(I)
    GT = np.squeeze(GT)
    Seg = np.squeeze(Seg)

    GTC = np.logical_and(GT, np.logical_not(ndimage.binary_erosion(GT)))
    SegC = np.logical_and(Seg, np.logical_not(ndimage.binary_erosion(Seg)))

    plt.figure(fig)
    maskedGT = np.ma.masked_where(GTC == 0, GTC)
    maskedSeg = np.ma.masked_where(SegC == 0, SegC)
    plt.imshow(I, cmap=cm.gray)
    plt.imshow(maskedGT, cmap=cm.jet, interpolation='none')
    plt.imshow(maskedSeg*100, cmap=cm.hsv, interpolation='none')

def run_session():
    coord = tf.train.Coordinator()
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.InteractiveSession(config=config)
    sess.run(init_op)
    tf.train.start_queue_runners(sess, coord=coord)
    return sess


def one_hot(x, depth):
    # workaround by name-name
    sparse_labels = tf.cast(tf.reshape(x, [-1, 1]),tf.int32)
    derived_size = tf.shape(sparse_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    outshape = tf.concat(axis=0, values=[tf.reshape(derived_size, [1]), tf.reshape(depth, [1])])
    return tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

def summary_tag_replace(summary_str, old, new=''):
    # Method to fix tb_1.x.x for validation
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary_str)
    for t, val in enumerate(summary_proto.value):
        summary_proto.value[t].__setattr__('tag', val.tag.replace(old, new))
    return summary_proto.SerializeToString()

