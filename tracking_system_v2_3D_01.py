import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import keras
import math
import os

plt.rcParams['animation.ffmpeg_path'] = "C:\\ProgramData\\Anaconda3\\pkgs\\ffmpeg-4.1.3-h6538335_0\\Library\\bin\\ffmpeg"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def data_iterator(raw_data, batch_size, win_size):
    (data_len, m) = raw_data.shape
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len, m], dtype=np.float32)
    for i in range(batch_size):
        data[i, :, :] = raw_data[batch_len * i : batch_len * (i + 1), :]

    epoch_size = (batch_len - 1) // win_size

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

# The code below avoids replication of data during its batching.
# Especially actual for TANECO prognostic CNN architecture with multiple data replication!!
    for j in range(epoch_size):
        x = data[:, j*win_size : (j + 1)*win_size, :]
        yield x
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_epochs(data, n, win_size, batch_size):
    for i in range(n):
        yield data_iterator(data, batch_size, win_size)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def spheric(p):
    teta, phi = p[:, 0 : 1], p[:, 1 : 2]
    x = tf.math.multiply(tf.sin(teta), tf.cos(phi))
    y = tf.math.multiply(tf.sin(teta), tf.sin(phi))
    z = tf.cos(teta)
    out = tf.concat([x, y, z], axis=1)
    return out
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def spheric_num(p):
    teta, phi = p[0], p[1]
    x = np.sin(teta)*np.cos(phi)
    y = np.sin(teta)*np.sin(phi)
    z = np.cos(teta)
    return [x, y, z]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def physics_r(tensors):
    ytm1 = tensors[0][:, :2]
    vtm1 = tensors[0][:, 2:]
    pow = tensors[1][:, 0 : 1]
    v = tensors[1][:, 1 : 3]
    vt = pow*v
    formula = tf.ones((1,))
    # formula = 1.0 - 0.25 * tf.norm(v - vtm1, axis=1) ** 2
    factor = tf.reshape(formula, [batch_size, 1])
    y1 = ytm1[:, 0] + dt_r*tf.math.multiply(vt[:, 0 : 1], factor)
    y2 = ytm1[:, 1] + dt_r*tf.math.multiply(vt[:, 1 : 2], factor)
    yt = tf.concat([y1, y2], axis=1)
    return yt, v
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def physics_w(tensors):
    ytm1 = tensors[0][:, :2]
    vtm1 = tensors[0][:, 2:]
    pow = tensors[1][:, 0 : 1]
    v = tensors[1][:, 1 : 3]
    vt = pow*v
    # formula = tf.ones((1,))
    formula = (1.0 - 0.5 * tf.norm(v - vtm1, axis=1))**2
    factor = tf.reshape(formula, [batch_size, 1])
    y1 = ytm1[:, 0 : 1] + dt_w*tf.math.multiply(vt[:, 0 : 1], factor)
    y2 = ytm1[:, 1 : 2] + dt_w*tf.math.multiply(vt[:, 1 : 2], factor)
    yt = tf.concat([y1, y2], axis=1)
    return yt, v
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def direct_metric(tensors):
    diff = tensors[0] - tensors[1]
    weight = 5
    out = weight*tf.norm(diff, axis=-1, keepdims=True)
    return out
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def inverse_metric(tensors):
    diff = tensors[0] - tensors[1]
    norm = tf.norm(diff, axis=-1, keepdims=True)
    # norm = tf.tanh(norm)
    eps = 0.1
    out = 1.0/(eps + tf.sqrt(norm))
    return out
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TrackingCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units=10, e_dim=3, inverse_metric=False, **kwargs):
        self._num_units = num_units
        self._out_size = (1, e_dim)
        self._inverse_metric = inverse_metric
        super(TrackingCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return (self._num_units, 2*p_dim, 1)

    @property
    def output_size(self):
        return self._out_size

    def __call__(self, inputs, state, scope=None):
        x3 = inputs
        state_nn_tm1, state_F_tm1, state_J_tm1 = state[0], state[1], state[2]
        nn_input = tf.concat([x3, state_F_tm1, state_J_tm1], axis=1)
        # gru_output, state_nn_t = tf.nn.rnn_cell.GRUCell(self._num_units)(inputs=nn_input, state=state_nn_tm1)
        gru_output, [state_nn_t] = tf.keras.layers.GRUCell(self._num_units)(inputs=nn_input, states=[state_nn_tm1])

        gru_output = tf.nn.tanh(gru_output)
        nn_output = tf.keras.layers.Dense(e_dim, activation='tanh')(gru_output)
        [pow, v] = tf.split(nn_output, [1, 2], axis=1)
        v = tf.nn.l2_normalize(v, axis=1)
        pow = tf.abs(pow)
        nn_output = tf.concat([pow, v], axis=1)

        if not self._inverse_metric:
            F_layer = tf.keras.layers.Lambda(physics_w)
            J_layer = tf.keras.layers.Lambda(direct_metric)
        else:
            F_layer = tf.keras.layers.Lambda(physics_r)
            J_layer = tf.keras.layers.Lambda(inverse_metric)

        y, v = F_layer([state_F_tm1, nn_output])
        state_F_t = tf.concat([y, v], axis=1)

        y3 = spheric(y)

        J = J_layer([x3, y3])
        state_J_t = J
        new_state = (state_nn_t, state_F_t, state_J_t)
        output = (J, y3)
        return output, new_state
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def reset_graph():
    global sess
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def tracking_graph(scope=None, num_units=10, e_dim=3, window_size=128, batch_size=1, learning_rate=1e-3, inverse_metric=False):

    with tf.name_scope(scope):
        cell = TrackingCell(num_units, e_dim, inverse_metric=inverse_metric)
        x = tf.placeholder(tf.float32, [batch_size, window_size, e_dim], name='input_3D_trajectorie')
        z = tf.zeros([batch_size, window_size, 1], tf.float32, name='output_zero_metric')
        init_state = cell.zero_state(batch_size, tf.float32)

#Depricated but working wrapper
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
    #This does not work
        # rnn_outputs, final_state = keras.layers.RNN(cell, return_sequences=True, return_state=True)(inputs=x, initial_state=init_state)

        J = tf.reshape(rnn_outputs[0], [-1, 1])
        y = rnn_outputs[1]
        # y = tf.reshape(rnn_outputs[1], [window_size, dim])
        z_reshaped = tf.reshape(z, [-1, 1])

        err = J - z_reshaped
        loss = tf.nn.l2_loss(err)
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        G = dict(x=x, J=J, y=y, init_state=init_state, final_state=final_state,
                 loss=loss, train_step=train_step)
    return G
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def train_networks(R, W, window_size=128, batch_size=1, watch=False, verbose=True, save=False):
    tf.set_random_seed(1111)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if watch:
            fig = plt.figure(figsize=[9, 9])
            axes = fig.gca(projection='3d')
            plot_sphere(radius, axes)
            axes.set_xlim(-radius, radius)
            axes.set_ylim(-radius, radius)
            axes.set_zlim(-radius, radius)
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.set_zticklabels([])

        training_losses = []
        position_r = []
        position_w = []

        steps = 0

        s1_r = np.reshape([init_position_r, init_velocity_r], [1, 2*p_dim])
        training_state_r = (np.zeros((1, num_units_r)), s1_r, np.zeros((1, 1)))
        training_loss_r = 0

        s1_w = np.reshape([init_position_w, init_velocity_w], [1, 2*p_dim])
        training_state_w = (np.zeros((1, num_units_w)), s1_w, np.zeros((1, 1)))
        w = np.tile(spheric_num(init_position_w), (batch_size, window_size, 1))
        training_loss_w = 0

        line_x, line_y = [], []
        for i in range(Len):
            steps += 1
            print(steps, '/', Len)
#Rabbit turn
            feed_dict={R['x']: w}
            feed_dict[R['init_state']] = training_state_r
            training_loss, training_state_r, _, _, r = sess.run([R['loss'], R['final_state'], R['train_step'],
                                                                 R['J'], R['y']], feed_dict)
            training_loss_r += training_loss
            position_r.append(r)
#Wolf turn
            feed_dict = {W['x']: r}
            feed_dict[W['init_state']] = training_state_w
            training_loss, training_state_w, _, _, w = sess.run([W['loss'], W['final_state'], W['train_step'],
                                                                 W['J'], W['y']], feed_dict)
            training_loss_w += training_loss
            position_w.append(w)

            if np.any(np.isnan(r)) or np.any(np.isnan(w)):
                print("Nan value of unit position!")
                return

            if watch:
                for j in range(batch_size):
                    if steps > 1:
                        line_x.remove()
                        line_y.remove()
                        axes.plot(r_old[:, 0], r_old[:, 1], r_old[:, 2], 'b', alpha=0.1, linewidth=2)
                        axes.plot(w_old[:, 0], w_old[:, 1], w_old[:, 2], 'r', alpha=0.1, linewidth=2)

                    line_x, = axes.plot(r[j, :, 0], r[j, :, 1], r[j, :, 2], 'b', linewidth=3)
                    line_y, = axes.plot(w[j, :, 0], w[j, :, 1], w[j, :, 2], 'r', linewidth=3)
                    r_old, w_old = r[j, :, :], w[j, :, :]
                    plt.pause(1e-17)
                    fig.canvas.draw()

        if watch:
            plt.close(fig)

        if verbose:
            print("Average training loss for Rabbit:", training_loss_r/steps)
            print("Average training loss for Wolf:", training_loss_w/steps)

        if save:
            saver = tf.train.Saver()
            saver.save(sess, model_dir + model_name)

    return training_losses, position_r, position_w
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def to_plot_all(x, y, speed_factor=3, fig_num=1, video_write=False, filename='my_movie.mp4'):

    x = np.vstack(x)
    x = np.reshape(x, [-1, e_dim])
    y = np.vstack(y)
    y = np.reshape(y, [-1, e_dim])

    fig = plt.figure(fig_num, figsize=[9, 9])
    axes = fig.gca(projection='3d')
    plot_sphere(radius, axes)
    axes.set_xlim(-radius, radius)
    axes.set_ylim(-radius, radius)
    axes.set_zlim(-radius, radius)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])

    if video_write:
        writer = animation.FFMpegWriter(fps=30)
        writer.setup(fig, filename)

    chunk_size = speed_factor*window_size
    len = x.shape[0]//chunk_size
    line_x, line_y = [], []

    for j in range(len):
        if j > 0:
            line_x.remove()
            line_y.remove()
            axes.plot(x[(j - 2) * chunk_size: (j - 1) * chunk_size, 0],
                      x[(j - 2) * chunk_size: (j - 1) * chunk_size, 1],
                      x[(j - 2) * chunk_size: (j - 1) * chunk_size, 2], 'b', alpha=0.1, linewidth=2)
            axes.plot(y[(j - 2) * chunk_size: (j - 1) * chunk_size, 0],
                      y[(j - 2) * chunk_size: (j - 1) * chunk_size, 1],
                      y[(j - 2) * chunk_size: (j - 1) * chunk_size, 2], 'r', alpha=0.1, linewidth=2)

        line_x, = axes.plot(x[(j - 1)*chunk_size : j*chunk_size, 0],
                            x[(j - 1)*chunk_size : j*chunk_size, 1],
                            x[(j - 1)*chunk_size : j*chunk_size, 2], 'b', linewidth=3)
        line_y, = axes.plot(y[(j - 1)*chunk_size : j*chunk_size, 0],
                            y[(j - 1)*chunk_size : j*chunk_size, 1],
                            y[(j - 1)*chunk_size : j*chunk_size, 2], 'r', linewidth=3)
        plt.pause(1e-17)
        fig.canvas.draw()

        if video_write:
            writer.grab_frame()

    if not video_write:
        plt.show()
    else:
        writer.cleanup()

    return []
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_sphere(radius, axes):
    ds = 0.1
    Phi = np.arange(-2 * math.pi, 2 * math.pi, ds)
    Teta = np.arange(-math.pi, math.pi, ds)
    Phi, Teta = np.meshgrid(Phi, Teta)
    X = radius*np.multiply(np.sin(Teta), np.cos(Phi))
    Y = radius*np.multiply(np.sin(Teta), np.sin(Phi))
    Z = radius*np.cos(Teta)
    ls = LightSource(azdeg=0, altdeg=65)
    rgb = ls.shade(Z, cmap=plt.get_cmap('bone'))
    surf = axes.plot_surface(X, Y, Z, linewidth=0, cmap='bone', alpha=0.1, facecolors=rgb)
    return []
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GPU/CPU usage for graph computations
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
gpuflag = 1
if gpuflag == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_dir = 'F:\\PythonFiles\\RecursiveNNTest\\Model\\'
model_name = 'WRmodel'
# model_topology = os.path.join(model_dir, model_name + '.json')
# model_weights = os.path.join(model_dir, model_name + '.hdf5')

#Sphere radius
radius = 1.0

#Initial positions in parametric space
init_position_r = [0.3, 0.3]
init_velocity_r = [0.0, 0.0]
dt_r = 0.002
num_units_r = 40

init_position_w = [0.9, -0.9]
init_velocity_w = [0, 0]
dt_w = 0.0022
num_units_w = 40

#Simulation time
Len = 200

p_dim = 2#do not change!
e_dim = 3#do not change!
window_size = 16
batch_size = 1#do not change!

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    reset_graph()

#Rabbit's creation
    R = tracking_graph(scope='rabbit', num_units=num_units_r, e_dim=e_dim, window_size=window_size, batch_size=batch_size, learning_rate=1e-3, inverse_metric=True)

#Wolf's creation
    W = tracking_graph(scope='wolf', num_units=num_units_w, e_dim=e_dim, window_size=window_size, batch_size=batch_size, learning_rate=1e-3)

#Statistics
    N_r = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('rabbit')])
    N_w = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('wolf')])
    N_all = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Number of the Wolf's trainable variables/all: ", N_w, "/", N_all)
    print("Number of the Rabbit's trainable variables/all: ", N_r,"/", N_all)

#Competition
    training_losses, position_r, position_w = train_networks(R, W, window_size=window_size, batch_size=batch_size, watch=True, save=True)
    # to_plot_all(x=position_r, y=position_w, speed_factor=1, fig_num=2, video_write=True, filename='02.mp4')

if __name__ == '__main__':
    main()