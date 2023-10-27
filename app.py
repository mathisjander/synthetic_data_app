import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

from timegan import timegan



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


import io
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE




def main():

    # title and description

    st.title("Synthetic Time Series Data Generator")

    st.write("""This is a proof of concept for using Streamlit to create a Synthetic Sensor Data Generator. 
             \nThe assessment should be considered indicative and further validation for a specific use case is likely required.""")
    st.markdown("""The TimeGAN implementation is based on a paper by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar: [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks), Neural Information Processing Systems (NeurIPS), 2019.""")


    st.divider()
    st.subheader("Upload the real data to be used for the synthetic data generation")
    st.write("""The data should be in csv format and have the following structure: 
             \nn * m
             \nWith n as the number of rows/obersvations in a fixed interval and m as the number of columns/sensor measurings for the given observation.""")
    
    # upload csv file

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write("Upload successful!")
        st.divider()
        st.subheader("Display of the uploaded data")

        st.write(f"Shape of the dataframe: {dataframe.shape[0]}, {dataframe.shape[1]}")
        st.dataframe(dataframe)

        # plot the uploaded data
        axes = dataframe.div(dataframe.iloc[0]).plot(subplots=True,
                               figsize=(14, 6),
                               layout=(3, 2),
                               title=dataframe.columns.tolist(),
                               legend=False,
                               rot=0,
                               lw=1, 
                               color='k')
        for ax in axes.flatten():
            ax.set_xlabel('')

        plt.gcf().tight_layout()
        sns.despine()

        # Display the plot in Streamlit
        st.pyplot(plt)
        st.divider()

        # TimeGAN params

         # define seq length
        st.subheader("Define the parameters for the synthetic data generation")

        seq_len = st.number_input('Define sequence length')
        seq_len = int(seq_len)
        st.write('The defined sequence length is ', seq_len)

        # define n sequences
        n_seq = st.number_input('Insert number of time series')
        n_seq = int(n_seq)
        st.write('The number of time series is ', n_seq)

        # define n sequences
        batch_size = st.number_input('Define batch size')
        batch_size = int(batch_size)
        st.write('The defined batch size is ', batch_size)

        # define hiddem dims
        hidden_dim = st.number_input('Define number of hidden dimensions')
        hidden_dim = int(hidden_dim)
        st.write('The defined number of hidden dimensions is ', hidden_dim)

        # define num layers
        num_layers = st.number_input('Define number of hidden layers')
        num_layers = int(num_layers)
        st.write('The defined number of hidden layers is ', num_layers)

        # define train steps
        train_steps = st.number_input('Define number of train steps')
        train_steps = int(train_steps)
        st.write('The defined number of train steps is ', train_steps)

        gamma = st.number_input('Define gamma', min_value=0.0, max_value=1.0, value=0.9, step=0.01)
        st.write('The defined gamma is ', np.round(gamma,2))


        ### timegan instance

        @st.cache_resource
        def train_model(dataframe, seq_len, n_seq, batch_size, hidden_dim, num_layers, train_steps, gamma):
            return timegan(dataframe,
                        seq_len,
                        n_seq,
                        batch_size,
                        hidden_dim,
                        num_layers,
                        train_steps,
                        gamma)
        
        synthetic_data, scaler, step_d_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = train_model(dataframe, seq_len, n_seq, batch_size, hidden_dim, num_layers, train_steps, gamma)
        
        # write the loss
        st.divider()
        st.subheader("Losses")
        st.write('The discriminator loss is ', step_d_loss.numpy())
        st.write('The unsupervised generator loss is ', step_g_loss_u.numpy())
        st.write('The supervised generator loss is ', step_g_loss_s.numpy())
        st.write('The generator moment loss is ', step_g_loss_v.numpy())
        st.write('The embedder loss is ', step_e_loss_t0.numpy())

        # download the model

        @st.cache_resource
        def pickle_model(_model):
            buf = io.BytesIO()
            pickle.dump(_model, buf)
            buf.seek(0)
            return buf

        model = pickle_model(synthetic_data)

        st.download_button(
            label="Download model as pickle",
            data=model,
            file_name='generator.pkl')

        st.divider()
        
        # generate data
        st.subheader("Generated synthetic data")

        # speficify number of windows
        n_windows = st.number_input('Define number of samples to generate')
        n_windows = int(n_windows)

        
        # define random input generator
        def make_random_data():
            while True:
                yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))

        
        # generate random series
        random_series = iter(tf.data.Dataset
                        .from_generator(make_random_data, output_types=tf.float32)
                        .batch(batch_size)
                        .repeat())

        # transform generated series into synthetic data
        generated_data = []
        for i in range(int(n_windows / batch_size)):
            Z_ = next(random_series)
            d = synthetic_data(Z_)
            generated_data.append(d)

        # write length of generated data
        st.write('The number of the generated batches is ', len(generated_data))

        generated_data = np.array(np.vstack(generated_data))
        st.write('The shape of the generated data is ', generated_data.shape)


        # implement tsne and pca for viz

        # Preprocess the real dataset:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataframe)

        real_data = []
        for i in range(len(dataframe) - seq_len):
            real_data.append(scaled_data[i:i + seq_len])

        real_data = real_data[:generated_data.shape[0]]

        # transform to 2d
        real_sample_2d = np.asarray(real_data).reshape(-1, seq_len)
        synthetic_sample_2d = np.asarray(generated_data).reshape(-1, seq_len)
        st.write('The shape of the generated data in 2D is ', synthetic_sample_2d.shape)

        # add button to download the generated data
        @st.cache_data
        def convert_array_to_csv(array, n_windows, seq_len):

            # convert 3D array to 2D array
            array2d = array.reshape(-1, n_seq)

            # convert array to dataframe
            df = pd.DataFrame(array2d)
            df.columns = dataframe.columns

            sample_ids = []

            for i in range(n_windows):
                sample_id = [i]*seq_len
                sample_ids.extend(sample_id)

            df['sample_id'] = sample_ids

            return df.to_csv().encode('utf-8')

        csv = convert_array_to_csv(generated_data, n_windows, seq_len)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='generated_data.csv',
            mime='text/csv')
        
        st.divider()

        # assessment section
        st.subheader("Assessment of Quality of Generated Data")
        
        # function for pca and tsne
        @st.cache_data
        def pca_tsne(real_sample, synthetic_sample):
            # PCA
            pca = PCA(n_components=2)
            pca.fit(real_sample)
            pca_real = (pd.DataFrame(pca.transform(real_sample))
                        .assign(Data='Real'))
            pca_synthetic = (pd.DataFrame(pca.transform(synthetic_sample))
                            .assign(Data='Synthetic'))
            pca_result = pca_real.append(pca_synthetic).rename(
                columns={0: '1st Component', 1: '2nd Component'})
            
            # t-SNE
            tsne_data = np.concatenate((real_sample,
                                synthetic_sample), axis=0)

            tsne = TSNE(n_components=2,
                        verbose=1,
                        perplexity=40)
            tsne_result = tsne.fit_transform(tsne_data)

            tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
            tsne_result.loc[n_windows*n_seq:, 'Data'] = 'Synthetic'

            return pca_result, tsne_result
        
        pca_results, tsne_results = pca_tsne(real_sample_2d, synthetic_sample_2d)
        

        # plot the results

        fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

        sns.scatterplot(x='1st Component', y='2nd Component', data=pca_results,
                        hue='Data', style='Data', ax=axes[0])
        sns.despine()
        axes[0].set_title('PCA Result')


        sns.scatterplot(x='X', y='Y',
                        data=tsne_results,
                        hue='Data', 
                        style='Data', 
                        ax=axes[1])
        sns.despine()
        for i in [0, 1]:
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        axes[1].set_title('t-SNE Result')
        fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', 
                    fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=.88)

        st.write('Assessing diversity: The distribution of the real and synthetic samples shoud be similar.')

        st.pyplot(plt)

        # implement classifier

        @st.cache_data
        def classification(real_data, generated_data, n_seq):

            # make sure data is np array
            real_data = np.array(real_data)
            generated_data = np.array(generated_data)
            
            # train test split
            n_series = real_data.shape[0]
            idx = np.arange(n_series)

            n_train = int(.8*n_series)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            train_data = np.vstack((real_data[train_idx], generated_data[train_idx]))
            test_data = np.vstack((real_data[test_idx], generated_data[test_idx]))
            
            n_train, n_test = len(train_idx), len(test_idx)
            train_labels = np.concatenate((np.ones(n_train),
                                        np.zeros(n_train)))
            test_labels = np.concatenate((np.ones(n_test),
                                        np.zeros(n_test)))
            
            # create classifier

            ts_classifier = Sequential([GRU(n_seq, input_shape=(seq_len, n_seq), name='GRU'),
                            Dense(1, activation='sigmoid', name='OUT')],
                           name='Time_Series_Classifier')
            
            ts_classifier.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[AUC(name='AUC'), 'accuracy'])
            
            ts_classifier.summary()

            # train classifier

            result = ts_classifier.fit(x=train_data,
                           y=train_labels,
                           validation_data=(test_data, test_labels),
                           epochs=250,
                           batch_size=128,
                           verbose=0)
            
            ts_classifier.evaluate(x=test_data, y=test_labels)
            history = pd.DataFrame(result.history)

            return history
        
        history = classification(real_data, generated_data, n_seq)

        # plot the results
        sns.set_style('white')
        fig, axes = plt.subplots(ncols=2, figsize=(14,4))
        history[['AUC', 'val_AUC']].rename(columns={'AUC': 'Train', 'val_AUC': 'Test'}).plot(ax=axes[1], 
                                                                                            title='ROC Area under the Curve',
                                                                                            style=['-', '--'],
                                                                                            xlim=(0, 250))
        history[['accuracy', 'val_accuracy']].rename(columns={'accuracy': 'Train', 'val_accuracy': 'Test'}).plot(ax=axes[0], 
                                                                                                            title='Accuracy',
                                                                                                            style=['-', '--'],
                                                                                                            xlim=(0, 250))
        for i in [0, 1]:
            axes[i].set_xlabel('Epoch')

        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y))) 
        axes[0].set_ylabel('Accuracy (%)')
        axes[1].set_ylabel('AUC')
        sns.despine()
        fig.suptitle('Assessing Fidelity: Time Series Classification Performance', fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=.85)

        st.write('Assessing fidelity: The classifier should not be able to distinguish between real and synthetic samples.')
        st.pyplot(plt)


        # implement tstr

        @st.cache_data
        def tstr(real_data, generated_data):

            # make sure data is np array
            real_data = np.array(real_data)
            generated_data = np.array(generated_data)

            # train test split
            n_series = real_data.shape[0]
            idx = np.arange(n_series)

            n_train = int(.8*n_series)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            real_train_data = real_data[train_idx, :seq_len-1, :]
            real_train_label = real_data[train_idx, -1, :]

            real_test_data = real_data[test_idx, :seq_len-1, :]
            real_test_label = real_data[test_idx, -1, :]

            synthetic_train = generated_data[:, :seq_len-1, :]
            synthetic_label = generated_data[:, -1, :]

            # define regression model used for tstr

            def get_model():
                model = Sequential([GRU(12, input_shape=(seq_len-1, n_seq)),
                                    Dense(3)])

                model.compile(optimizer=Adam(),
                            loss=MeanAbsoluteError(name='MAE'))
                return model
            
            # train on synthetic data

            ts_regression = get_model()
            synthetic_result = ts_regression.fit(x=synthetic_train,
                                                y=synthetic_label,
                                                validation_data=(
                                                    real_test_data, 
                                                    real_test_label),
                                                epochs=100,
                                                batch_size=128,
                                                verbose=0)
            
            # train on real data
            ts_regression = get_model()
            real_result = ts_regression.fit(x=real_train_data,
                                            y=real_train_label,
                                            validation_data=(
                                                real_test_data, 
                                                real_test_label),
                                            epochs=100,
                                            batch_size=128,
                                            verbose=0)
            
            synthetic_result = pd.DataFrame(synthetic_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
            real_result = pd.DataFrame(real_result.history).rename(columns={'loss': 'Train', 'val_loss': 'Test'})
            
            return synthetic_result, real_result
        
        synthetic_result, real_result = tstr(real_data, generated_data)

        # plot the results

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
        synthetic_result.plot(ax=axes[0], title='Train on Synthetic, Test on Real', logy=True, xlim=(0, 100))
        real_result.plot(ax=axes[1], title='Train on Real, Test on Real', logy=True, xlim=(0, 100))
        for i in [0, 1]:
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Mean Absolute Error (log scale)')

        sns.despine()
        fig.suptitle('Assessing Usefulness: Time Series Prediction Performance', fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=.85)

        st.write('Assessing usefulness: The synthetic data should be as useful as the real data for training a predictor.')
        st.pyplot(plt)



if __name__ == "__main__":
    main()

