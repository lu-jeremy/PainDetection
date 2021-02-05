from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model
from keras.callbacks import Callback, EarlyStopping

import matplotlib.pyplot as plt
import sys
import numpy as np
import gc

from vis.visualization import visualize_saliency
from vis.utils import utils

from Preprocessing import preprocessing


class PainNet:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessing(two_dimensional=True)

        self.model = self.generate_model_2d()
        # self.model_path = r'models\PainNet2DReformattedImagesRegression.h5'
        # self.model = load_model(self.model_path)

        self.model.summary()

    @staticmethod
    def generate_model_3d():
        cnn = Sequential()

        cnn.add(Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(209, 42, 80, 80),
                       data_format='channels_first', activation='relu', kernel_regularizer=l2(0.01)))
        cnn.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(.2))

        for filter_num in range(0, 4):
            # filters double every set of layers
            cnn.add(Conv3D(filters=64*2**filter_num, kernel_size=(3, 3, 3), activation='relu', padding='same'))
            cnn.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
            cnn.add(BatchNormalization())
            cnn.add(Dropout(.2))

        cnn.add(Flatten())

        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dropout(.2))
        cnn.add(Dense(209, activation='sigmoid'))

        cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
        cnn.summary()

        return cnn

    @staticmethod
    def generate_model_2d():
        cnn = Sequential()

        cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(1, 80, 80),
                       data_format='channels_first', activation='relu', kernel_regularizer=l2(0.01)))
        cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(.25))

        for filter_num in range(0, 4):
            cnn.add(Conv2D(filters=64*2**filter_num, kernel_size=(3, 3), activation='relu', padding='same'))
            cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            cnn.add(BatchNormalization())

        cnn.add(Flatten())

        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dense(1, activation='linear'))

        cnn.compile(optimizer=Adam(), loss='mean_squared_logarithmic_error')
        cnn.summary()

        return cnn

    def fit(self):
        early_stop = EarlyStopping(monitor='val_loss', mode='max', patience=15, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train, batch_size=32, validation_data=(self.X_test, self.y_test),
                                 epochs=1, callbacks=[early_stop, self.GarbageCollection()], verbose=1, shuffle=True)

        self.model.save(r'models\PainNet2DReformattedImagesRegression.h5')

        self.show_history(history)


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
        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        #
        # plt.plot(epoch_count, acc, 'c-')
        # plt.plot(epoch_count, val_acc, 'g--')
        # plt.legend(['Accuracy', 'Validation Accuracy'])
        # plt.xlabel('Epoch')
        # plt.ylabel('Acc')
        # plt.show()

    def vis_saliency(self):
        layer_index = utils.find_layer_idx(self.model, 'conv2d_20')

        # for i in np.arange(2, 4):
        input_image = self.X_test[100, :, :]
        #     input_class = np.argmax(self.y_test[i])

        grads = visualize_saliency(self.model, layer_index, filter_indices=[10], seed_input=input_image)

        # from visual import Visual
        # for one in range(41):
        #     slice_0 = input_image[one, :, :]
        #     slice_1 = input_image[:, one, :]
        #     slice_2 = input_image[:, :, one]
        #
        #     Visual().show_slices([slice_0, slice_1, slice_2])
        #     plt.show()

        input_image = input_image[:, 22, :]
        # Plot with 'jet' colormap to visualize as a heatmap.
        print('grads shape:', grads.shape)
        print('image shape:', input_image.shape)

        plt.imshow(grads, cmap='jet')
        plt.imshow(input_image, cmap='gray', alpha=.5)
        plt.show()

    def evalute(self):
        evaluation = self.model.evaluate(self.X_test, self.y_test)
        print(evaluation)

    def predict(self):
        print('Prediction:', self.model.predict(self.X_test, batch_size=32))

    class GarbageCollection(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()


if __name__ == '__main__':
    cnn = PainNet()
    cnn.fit()
    cnn.evalute()
    cnn.predict()
    # cnn.vis_saliency()
