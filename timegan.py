"""
Adapted from [cite algo trading book here]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model


def timegan(df, seq_len, n_seq, batch_size, hidden_dim, num_layers, train_steps, gamma=1):

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')

    results_path = Path('time_gan')
    if not results_path.exists():
        results_path.mkdir()

    experiment = 0

    log_dir = results_path / f'experiment_{experiment:02}'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    ## 1 Preprocessing
    # 1.1 scale the data

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df).astype(np.float32)

    # 1.2 create rolling window sequences

    data = []
    for i in range(len(df) - seq_len):
        data.append(scaled_data[i:i + seq_len])

    n_windows = len(data)

    # 1.3 Create tf.data.Dataset

    real_series = (tf.data.Dataset
               .from_tensor_slices(data)
               .shuffle(buffer_size=n_windows)
               .batch(batch_size))
    real_series_iter = iter(real_series.repeat())

    # 1.4 set up random series generator

    def make_random_data():
        while True:
            yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))

    random_series = iter(tf.data.Dataset
                     .from_generator(make_random_data, output_types=tf.float32)
                     .batch(batch_size)
                     .repeat())
    
    ## 2 TimeGAN components
    # 2.1 Network parameters

    hidden_dim   = hidden_dim
    num_layers   = num_layers

    # 2.2 set up logger
    writer = tf.summary.create_file_writer(log_dir.as_posix())

    # 2.3 input placeholders
    X = Input(shape=[seq_len, n_seq], name='RealData')
    Z = Input(shape=[seq_len, n_seq], name='RandomData')

    # 2.4 RNN block generator
    def make_rnn(n_layers, hidden_units, output_units, name):
        return Sequential([GRU(units=hidden_units,
                            return_sequences=True,
                            name=f'GRU_{i + 1}') for i in range(n_layers)] +
                        [Dense(units=output_units,
                                activation='sigmoid',
                                name='OUT')], name=name)
    
    # 2.5 Embedder and Recovery

    embedder = make_rnn(n_layers=3, 
                    hidden_units=hidden_dim, 
                    output_units=hidden_dim, 
                    name='Embedder')
    recovery = make_rnn(n_layers=3, 
                        hidden_units=hidden_dim, 
                        output_units=n_seq, 
                        name='Recovery')
    
    # 2.6 Generator and Discriminator

    generator = make_rnn(n_layers=3, 
                     hidden_units=hidden_dim, 
                     output_units=hidden_dim, 
                     name='Generator')
    discriminator = make_rnn(n_layers=3, 
                            hidden_units=hidden_dim, 
                            output_units=1, 
                            name='Discriminator')
    supervisor = make_rnn(n_layers=2, 
                        hidden_units=hidden_dim, 
                        output_units=hidden_dim, 
                        name='Supervisor')

    # 3 TimeGAN Training
    # 3.1 Settings

    train_steps = train_steps
    gamma = gamma

    # 3.2 Generic loss functions

    mse = MeanSquaredError()
    bce = BinaryCrossentropy()

    # 3.3 Phase 1: Autoencoder training

    H = embedder(X)
    X_tilde = recovery(H)

    autoencoder = Model(inputs=X,
                        outputs=X_tilde,
                        name='Autoencoder')
    
    autoencoder_optimizer = Adam()

    @tf.function
    def train_autoencoder_init(x):
        with tf.GradientTape() as tape:
            x_tilde = autoencoder(x)
            embedding_loss_t0 = mse(x, x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)
    
    for step in tqdm(range(train_steps)):
        X_ = next(real_series_iter)
        step_e_loss_t0 = train_autoencoder_init(X_)
        with writer.as_default():
            tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)

    # 3.4 Phase 2: Supervised training

    supervisor_optimizer = Adam()

    @tf.function
    def train_supervisor(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        var_list = supervisor.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s
    
    for step in tqdm(range(train_steps)):
        X_ = next(real_series_iter)
        step_g_loss_s = train_supervisor(X_)
        with writer.as_default():
            tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)

    # 3.5 Phase 3: Joint Training

    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    Y_fake = discriminator(H_hat)

    adversarial_supervised = Model(inputs=Z,
                                outputs=Y_fake,
                                name='AdversarialNetSupervised')

    Y_fake_e = discriminator(E_hat)

    adversarial_emb = Model(inputs=Z,
                        outputs=Y_fake_e,
                        name='AdversarialNet')
    
    X_hat = recovery(H_hat)
    synthetic_data = Model(inputs=Z,
                        outputs=X_hat,
                        name='SyntheticData')
    
    def get_generator_moment_loss(y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var
    
    Y_real = discriminator(H)
    discriminator_model = Model(inputs=X,
                                outputs=Y_real,
                                name='DiscriminatorReal')
    
    generator_optimizer = Adam()
    discriminator_optimizer = Adam()
    embedding_optimizer = Adam()

    @tf.function
    def train_generator(x, z):
        with tf.GradientTape() as tape:
            y_fake = adversarial_supervised(z)
            generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
                                            y_pred=y_fake)

            y_fake_e = adversarial_emb(z)
            generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
                                                y_pred=y_fake_e)
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_hat = synthetic_data(z)
            generator_moment_loss = get_generator_moment_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                            generator_loss_unsupervised_e +
                            100 * tf.sqrt(generator_loss_supervised) +
                            100 * generator_moment_loss)

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        generator_optimizer.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
    

    @tf.function
    def train_embedder(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_tilde = autoencoder(x)
            embedding_loss_t0 = mse(x, x_tilde)
            e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        embedding_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)
    
    @tf.function
    def get_discriminator_loss(x, z):
        y_real = discriminator_model(x)
        discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
                                    y_pred=y_real)

        y_fake = adversarial_supervised(z)
        discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
                                    y_pred=y_fake)

        y_fake_e = adversarial_emb(z)
        discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
                                        y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                gamma * discriminator_loss_fake_e)
    
    @tf.function
    def train_discriminator(x, z):
        with tf.GradientTape() as tape:
            discriminator_loss = get_discriminator_loss(x, z)

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        discriminator_optimizer.apply_gradients(zip(gradients, var_list))
        return discriminator_loss
    
    # 3.6 Training loop

    step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
    for step in range(train_steps):
        # Train generator (twice as often as discriminator)
        for kk in range(2):
            X_ = next(real_series_iter)
            Z_ = next(random_series)

            # Train generator
            step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
            # Train embedder
            step_e_loss_t0 = train_embedder(X_)

        X_ = next(real_series_iter)
        Z_ = next(random_series)
        step_d_loss = get_discriminator_loss(X_, Z_)
        if step_d_loss > 0.15:
            step_d_loss = train_discriminator(X_, Z_)

        if step % 1000 == 0:
            print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

        with writer.as_default():
            tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
            tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
            tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
            tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
            tf.summary.scalar('D Loss', step_d_loss, step=step)

    return synthetic_data, scaler, step_d_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0









