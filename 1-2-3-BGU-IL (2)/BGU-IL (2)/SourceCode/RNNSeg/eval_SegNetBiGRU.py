import os
import time

import numpy as np
import scipy.misc
import tensorflow as tf

from RNNSeg.LSTM_Network import BiGRUNetwork
from RNNSeg.Params import ParamsEvalBiGRU


def run_net():

    # Data input
    data_provider = params.data_provider

    with tf.name_scope('Data'):
        image_seq_fw, filename_seq_fw, image_seq_bw, filename_seq_bw = data_provider.get_sequence(params.seq_length)
    image_seq_fw_bw = [tf.concat([fw, bw], axis=0) for fw, bw in zip(image_seq_fw, image_seq_bw[::-1])]
    filename_seq_bw = filename_seq_bw[::-1]
    # Build Network Graph
    net_fw = BiGRUNetwork()
    net_bw = BiGRUNetwork()

    with tf.device('/gpu:0'):
        with tf.name_scope('run_tower'):
            image_seq_norm_fw = [tf.div(tf.subtract(im, params.norm),
                                        params.norm) for im in image_seq_fw]
            image_seq_norm_bw = [tf.div(tf.subtract(im, params.norm),
                                        params.norm) for im in image_seq_bw[::-1]]
            with tf.variable_scope('net'):
                _ = net_fw.build(image_seq_norm_fw, phase_train=False, net_params=params.net_params)
            with tf.variable_scope('net', reuse=True):
                _ = net_bw.build(image_seq_norm_bw, phase_train=False, net_params=params.net_params)
            fw_outputs = net_fw.fw_outputs
            bw_outputs = net_bw.bw_outputs

        fw_ph = tf.placeholder(tf.float32, params.image_size + (3,), 'fw_placeholder')
        bw_ph = tf.placeholder(tf.float32, params.image_size + (3,), 'bw_placeholder')
        merged = tf.add(fw_ph, bw_ph)
        final_out = tf.nn.softmax(merged)

    saver = tf.train.Saver(var_list=tf.global_variables())
    init_op = tf.group(tf.local_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if params.load_checkpoint:
            saver.restore(sess, params.load_checkpoint_path)

        threads = tf.train.start_queue_runners(sess, coord=coord)
        elapsed_time = 0.
        end_time = 0.
        other_time = 0.
        options = tf.RunOptions()
        feed_dict = {}
        loop = True
        all_filenames = []
        sigout =  [tf.nn.softmax(tf.transpose(o,(0,2,3,1))) for o in net_fw.outputs]
        t = 0
        while loop:
            try:
                t+=1
                start_time = time.time()
                other_time += start_time - end_time
                fetch_out = sess.run([fw_outputs, bw_outputs, net_fw.states[-1], net_bw.states_back[-1],
                                      filename_seq_fw, filename_seq_bw], options=options,
                                     feed_dict=feed_dict)
                seg_seq_out_fw, seg_seq_out_bw, states_fw, states_bw, file_names_fw, file_names_bw = fetch_out
                end_time = time.time()
                elapsed_time += end_time-start_time

                for state_ph, last_state in zip(net_fw.states[0], states_fw):
                    feed_dict[state_ph] = last_state

                for state_ph, last_state in zip(net_bw.states_back[0], states_bw):
                    feed_dict[state_ph] = last_state

                if not params.dry_run:
                    out_dir = params.experiment_tmp_fw_dir
                    for file_name, image_seg in zip(file_names_fw, seg_seq_out_fw):
                        file_name = file_name.decode('utf-8')
                        fw_squeeze = np.squeeze(image_seg)
                        fw_squeeze = fw_squeeze.transpose([1, 2, 0])
                        fw_filename = os.path.join(out_dir, os.path.basename(file_name))
                        np.save(fw_filename,fw_squeeze)
                        all_filenames.append(os.path.basename(file_name)+'.npy')
                        print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name))))
                    out_dir = params.experiment_tmp_bw_dir
                    for file_name, image_seg in zip(file_names_bw, seg_seq_out_bw):
                        file_name = file_name.decode('utf-8')
                        bw_squeeze = np.squeeze(image_seg)
                        bw_squeeze = bw_squeeze.transpose([1, 2, 0])
                        bw_filename = os.path.join(out_dir, os.path.basename(file_name))
                        np.save(bw_filename, bw_squeeze)
                        print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name))))

            except (ValueError, RuntimeError, KeyboardInterrupt, tf.errors.OutOfRangeError):

                coord.request_stop()
                coord.join(threads)
                loop = False
        if not params.dry_run:
            out_dir = params.experiment_out_dir
            for file_name in all_filenames:
                fw_logits = np.load(os.path.join(params.experiment_tmp_fw_dir, file_name))
                bw_logits = np.load(os.path.join(params.experiment_tmp_bw_dir, file_name))
                feed_dict = {bw_ph: bw_logits, fw_ph: fw_logits}
                seg_out = sess.run(final_out, feed_dict)
                seg_out = (seg_out*255).astype(np.uint8)
                scipy.misc.toimage(seg_out, cmin=0.0,
                                   cmax=255.).save(os.path.join(out_dir, os.path.basename(file_name[:-4])))
                print("Saved File: {}".format(os.path.join(out_dir, os.path.basename(file_name[:-4]))))


        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    params = ParamsEvalBiGRU()
    run_net()
