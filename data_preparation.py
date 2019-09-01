import numpy
from keras.preprocessing.image import load_img, img_to_array


def generate_arrays(images, masks, x_unet, y_unet, image_path, mask_path, quick_test=False):
    images_train = numpy.zeros((1, x_unet * y_unet))
    mask_train = numpy.zeros((1, x_unet * y_unet))

    for id in range(len(images)):
        np_image = conver_image_to_np_array(images[id], image_path, x_unet, y_unet)
        np_mask = conver_image_to_np_array(masks[id], mask_path, x_unet, y_unet)

        images_train = numpy.vstack((images_train, np_image))
        mask_train = numpy.vstack((mask_train, np_mask))

    images_train = numpy.delete(images_train, 0, 0)
    mask_train = numpy.delete(mask_train, 0, 0)
    images_train = images_train.reshape(images_train.shape[0], x_unet, y_unet, 1)
    mask_train = mask_train.reshape(mask_train.shape[0], x_unet, y_unet, 1)
    if quick_test:
        return numpy.delete(images_train, range(int(len(images) * 0.975)), 0), numpy.delete(mask_train,
                                                                                          range(
                                                                                              int(len(images) * 0.975)),
                                                                                          0)
    else:
        return images_train, mask_train


def conver_image_to_np_array(image, path, x_unet, y_unet):
    image = load_img(path + "/" + image)
    image.thumbnail((x_unet, y_unet))
    image = image.convert('L')

    np_image = img_to_array(image)
    np_image = np_image.reshape(x_unet * y_unet, )
    np_image = np_image

    return np_image/255
