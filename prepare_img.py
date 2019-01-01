from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np



def get_img_generator():
    """
    from keras.preprocessing.image.ImageDataGenerator prepare other image
    :return:
    """
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest'
    )


if __name__ == "__main__":
    img_prep = get_img_generator()

    img_w = 200
    img_h = 200
    img_dir = './'
    batch_size = 8

    img_gen = img_prep.flow_from_directory(
        img_dir,
        target_size=(img_w, img_h),
        save_to_dir='/home/out',
        class_mode='binary',
        save_prefix='N',
        save_format='jpeg',
        batch_size=batch_size
    )

    for inputs, outputs in img_gen:
        img = Image.fromarray(outputs)
        img.save('my.png')
        # print(outputs)
