from keras import backend as K
import tensorflow as tf

from MaskRCNN import Mask


if __name__ == '__main__':
    model=Mask()
    model.model.keras_model.summary()
    model.save_model()