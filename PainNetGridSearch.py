from keras.models import Sequential
from keras.layers import LSTM, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, LeakyReLU, Flatten, Dense, TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, Callback

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import sys
import gc
from time import time

from Preprocessing import preprocessing


class PainNet:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessing(two_dimensional=True)

        self.start = time()
        scitkit_wrapper_model = KerasClassifier(build_fn=self.generate_model_2d, verbose=1)
        self.model = self.grid_search(scitkit_wrapper_model)

    @staticmethod
    def generate_model_3d(pool_type='max', dropout_rate=0):
        cnn = Sequential()

        cnn.add(Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(209, 42, 80, 80),
                       data_format='channels_first', activation='relu', kernel_regularizer=l2(0.01)))
        if pool_type == 'max':
            cnn.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        else:
            cnn.add(AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        cnn.add(BatchNormalization())
        if dropout_rate != 0:
            cnn.add(Dropout(dropout_rate))

        for filter_num in range(0, 4):
            # filters double every set of layers
            cnn.add(Conv3D(filters=64*2**filter_num, kernel_size=(3, 3, 3), activation='relu', padding='same'))
            if pool_type == 'max':
                cnn.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
            else:
                cnn.add(AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
            cnn.add(BatchNormalization())
            if dropout_rate != 0:
                cnn.add(Dropout(dropout_rate))

        cnn.add(Flatten())

        cnn.add(Dense(512, activation='relu'))
        if dropout_rate != 0:
            cnn.add(Dropout(dropout_rate))
        cnn.add(Dense(209, activation='sigmoid'))

        cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
        cnn.summary()

        return cnn

    @staticmethod
    def generate_model_2d(pool_type='max', dropout_rate=0):
        cnn = Sequential()

        cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(42, 80, 80),
                       data_format='channels_first', activation='relu', kernel_regularizer=l2(0.01)))
        if pool_type == 'max':
            cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            cnn.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        cnn.add(BatchNormalization())
        if dropout_rate != 0:
            cnn.add(Dropout(dropout_rate))

        for filter_num in range(0, 4):
            # filters double every set of layers
            cnn.add(Conv2D(filters=64*2**filter_num, kernel_size=(3, 3), activation='relu', padding='same'))
            if pool_type == 'max':
                cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            else:
                cnn.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
            cnn.add(BatchNormalization())
            if dropout_rate != 0:
                cnn.add(Dropout(dropout_rate))

        cnn.add(Flatten())

        cnn.add(Dense(512, activation='relu'))
        if dropout_rate != 0:
            cnn.add(Dropout(dropout_rate))
        cnn.add(Dense(1, activation='sigmoid'))

        cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
        cnn.summary()

        return cnn

    @staticmethod
    def grid_search(model):
        # don't put too many parameters to conserve time
        # put in more for now
        parameters = [{
            'pool_type': ['average'],
            'epochs': [5],
            'dropout_rate': [0.10]
        }]

        clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy')

        return clf

    def show_grid_search_performance(self, results):
        print('duration of grid search processing = {:.0f} sec'.format(time() - self.start))

        print('Best score = {:.4f} using {}'.format(results.best_score_, results.best_params_))
        all_means = results.cv_results_['mean_test_score']
        all_stds = results.cv_results_['std_test_score']
        all_params = results.cv_results_['params']

        for mean, stds, params in zip(all_means, all_stds, all_params):
            print('mean_test_accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stds, params))

    def fit(self):
        early_stop = EarlyStopping(monitor='val_loss', mode='max', patience=5, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train, batch_size=10, validation_data=(self.X_test, self.y_test),
                                 callbacks=[early_stop, self.GarbageCollection()], verbose=1, shuffle=True)

        self.show_grid_search_performance(history)

        best_model = history.best_estimator_.model
        best_model.save(r'models\grid_search_best_model.h5')

    @staticmethod
    def show_history(history):
        # show training/testing loss over time
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)

        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        # show accuracy and validation accuracy over time
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        plt.plot(epoch_count, acc, 'c-')
        plt.plot(epoch_count, val_acc, 'g--')
        plt.legend(['Accuracy', 'Validation Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.show()

    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype("int32")

    def surrogate(self):
        y_cnn_train = self.predict(self.X_train)
        y_cnn_test = self.predict(self.X_test)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], -1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], -1))
        lr = LogisticRegression()
        lr.fit(self.X_train, y_cnn_train)

        acc = lr.score(self.X_test, y_cnn_test)
        print(acc)

        coef = lr.coef_
        coef = coef.reshape(80, 80)

        plt.imshow(coef)

        plt.show()

    class GarbageCollection(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()


if __name__ == '__main__':
    cnn = PainNet()
    cnn.fit()



