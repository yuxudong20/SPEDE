import numpy as np
import tensorflow.compat.v1 as tf

_ALLOW_KEY = ['state', 'diff_state', 'action']


def init_whitening_stats(key, size):
    whitening_stats = {}
    whitening_stats[key] = {'mean': tf.Variable(tf.zeros([size])), 'variance': tf.Variable(tf.ones([size])), 'step': tf.Variable(tf.zeros([1])+0.01),
                            'square_sum': tf.Variable(tf.zeros([size])+0.01), 'sum': tf.Variable(tf.zeros([size])), 'std': tf.Variable(tf.ones([size]))}
    return whitening_stats

@tf.function
def update_whitening_stats(whitening_stats, rollout_data, key):
    if key == 'state':
        i_data = rollout_data['s1']
    elif key == 'action':
        i_data = rollout_data['a']
    else:
        assert key == 'diff_state'
        i_data = rollout_data['s2'] - rollout_data['s1']

    new_sum = tf.reduce_sum(i_data, axis=0)
    new_sq_sum = tf.reduce_sum(tf.math.square(i_data), axis=0)
    new_step_sum = tf.expand_dims(i_data.shape[0], axis=0)
    new_sum = tf.cast(new_sum, dtype=tf.float32)
    new_sq_sum = tf.cast(new_sq_sum, dtype=tf.float32)
    new_step_sum = tf.cast(new_step_sum, dtype=tf.float32)

    # update the whitening info
    whitening_stats[key]['step'].assign_add(new_step_sum)
    whitening_stats[key]['sum'].assign_add(new_sum)
    whitening_stats[key]['square_sum'].assign_add(new_sq_sum)
    whitening_stats[key]['mean'].assign(whitening_stats[key]['sum'] / whitening_stats[key]['step'])
    whitening_stats[key]['variance'].assign(tf.math.maximum(
        whitening_stats[key]['square_sum'] / whitening_stats[key]['step'] -
        tf.math.square(whitening_stats[key]['mean']), tf.constant([1e-2])
    ))
    whitening_stats[key]['std'].assign((whitening_stats[key]['variance'] + 1e-6) ** .5)


def add_whitening_operator(whitening_operator, whitening_variable, name, size):

    with tf.variable_scope('whitening_' + name):
        whitening_operator[name + '_mean'] = tf.Variable(
            np.zeros([1, size], np.float32),
            name=name + "_mean", trainable=False
        )
        whitening_operator[name + '_std'] = tf.Variable(
            np.ones([1, size], np.float32),
            name=name + "_std", trainable=False
        )
        whitening_variable.append(whitening_operator[name + '_mean'])
        whitening_variable.append(whitening_operator[name + '_std'])

        # the reset placeholders
        whitening_operator[name + '_mean_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_mean_ph'
        )
        whitening_operator[name + '_std_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_std_ph'
        )

        # the tensorflow operators
        whitening_operator[name + '_mean_op'] = \
            whitening_operator[name + '_mean'].assign(
                whitening_operator[name + '_mean_ph']
        )

        whitening_operator[name + '_std_op'] = \
            whitening_operator[name + '_std'].assign(
                whitening_operator[name + '_std_ph']
        )


def set_whitening_var(session, whitening_operator, whitening_stats, key_list):

    for i_key in key_list:
        for i_item in ['mean', 'std']:
            session.run(
                whitening_operator[i_key + '_' + i_item + '_op'],
                feed_dict={whitening_operator[i_key + '_' + i_item + '_ph']:
                           np.reshape(whitening_stats[i_key][i_item], [1, -1])}
            )


def append_normalized_data_dict(data_dict, whitening_stats,
                                target=['start_state', 'diff_state',
                                        'end_state']):
    data_dict['n_start_state'] = \
        (data_dict['start_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_end_state'] = \
        (data_dict['end_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_diff_state'] = \
        (data_dict['end_state'] - data_dict['start_state'] -
         whitening_stats['diff_state']['mean']) / \
        whitening_stats['diff_state']['std']
    data_dict['diff_state'] = \
        data_dict['end_state'] - data_dict['start_state']
