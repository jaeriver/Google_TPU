from keras.applications.imagenet_utils import obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.utils.layer_utils import get_source_inputs

'''Google MobileNet model for Keras.
# Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
'''

def MobileNet(input_tensor=None, input_shape=None, alpha=1, shallow=False, num_classes=1000):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        
        # Arguments
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 244)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 96.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
        # Returns
            A Keras model instance.

        """

    input_shape = obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    def dw(x,dw_pad,conv_f,conv_st):
            x = DepthwiseConv2D(kernel_size=(3,3),padding = 'same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x) 
            x = Conv2D(filters= conv_f,kernel_size=(1,1),strides=conv_st,padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
    x = Conv2D(filters=int(32*alph),kernel_size=(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = dw(x,'same',int(64*alph),1)
    x = dw(x,'valid',int(128*alph),2)
    x = dw(x,'same',int(128*alph),1)
    x = dw(x,'same',int(256*alph),2)
    x = dw(x,'same',int(256*alph),1)
    x = dw(x,'valid',int(512*alph),2)
    for i in range(5):
        x = dw(x,'same',int(512*alph),1) 
    x = dw(x,'valid',int(1024*alph),2)
    x = dw(x,'same',int(1024*alph),1)

    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='mobilenet')

    return model


if __name__ == '__main__':
    m = MobileNet()
    print("model ready")
