from keras.models import *
from keras.layers import *
from keras.optimizers import *

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet(x_unet, y_unet, filter_no, optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy']):
    inputs = Input((x_unet, y_unet, 1))

    conv1 = Conv2D(filter_no, 3, strides=(1, 1), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filter_no, 3, strides=(1, 1), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = Conv2D(filter_no * 2, 3, strides=(1, 1), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filter_no * 2, 3, strides=(1, 1), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv3 = Conv2D(filter_no * 4, 3, strides=(1, 1), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filter_no * 4, 3, strides=(1, 1), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    conv4 = Conv2D(filter_no * 8, 3, strides=(1, 1), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filter_no * 8, 3, strides=(1, 1), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
    conv5 = Conv2D(filter_no * 16, 3, strides=(1, 1), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filter_no * 16, 3, strides=(1, 1), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    merge1 = concatenate([conv4, up1], axis=3)
    conv6 = Conv2D(filter_no * 8, 3, strides=(1, 1), activation='relu', padding='same')(merge1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(filter_no * 8, 3, strides=(1, 1), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    merge2 = concatenate([conv3, up2], axis=3)
    conv7 = Conv2D(filter_no * 4, 3, strides=(1, 1), activation='relu', padding='same')(merge2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(filter_no * 4, 3, strides=(1, 1), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    merge3 = concatenate([conv2, up3], axis=3)
    conv8 = Conv2D(filter_no * 2, 3, strides=(1, 1), activation='relu', padding='same')(merge3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(filter_no * 2, 3, strides=(1, 1), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    merge4 = concatenate([conv1, up4], axis=3)
    conv9 = Conv2D(filter_no, 3, strides=(1, 1), activation='relu', padding='same')(merge4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(filter_no, 3, strides=(1, 1), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, strides=(1, 1), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv9)

    if loss == 'dice_coef':
        model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # model.summary()
    return model
