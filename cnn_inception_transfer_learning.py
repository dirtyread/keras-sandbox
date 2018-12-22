# inspired by https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# data from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data.
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    print("start")

    dogs_train_num = 1000
    cats_train_num = 1000
    dogs_validate_num = 400
    cats_validate_num = 400

    Image_width, Image_height = 299, 299
    Number_FC_Neurons = 1024

    train_dir = './data/train'
    validate_dir = './data/validate'
    num_classes = 2
    num_train_samples = dogs_train_num + cats_train_num
    num_validate_samples = dogs_validate_num + cats_validate_num
    num_epoch = 2
    batch_size = 32

    train_img_prep = get_img_generator()
    test_img_prep = get_img_generator()

    train_gen = train_img_prep.flow_from_directory(
        train_dir,
        target_size=(Image_width, Image_height),
        batch_size=batch_size
    )

    valid_gen = test_img_prep.flow_from_directory(
        validate_dir,
        target_size=(Image_width, Image_height),
        batch_size=batch_size
    )

    # START model
    # model without last FC layer - include_top=False
    inceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)

    # create out layers in the new classification prediction
    x = inceptionV3_base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(Number_FC_Neurons, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inceptionV3_base_model.input, outputs=predictions)

    # disp. model summary
    #print(model.summary())

    # start transfer learning - disable change values in layer.trainable
    # for layer in inceptionV3_base_model.layers:
    #     layer.trainable = False
    #
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # history_transfer_learning = model.fit_generator(
    #     train_gen,
    #     epochs=num_epoch,
    #     steps_per_epoch=num_train_samples // batch_size,
    #     validation_data=valid_gen,
    #     validation_steps=num_validate_samples // batch_size,
    #     class_weight='auto')
    #
    # model.save('inception3-transfer-learning.model')

    # transfer learning with fine-tuning - train only few layers
    num_layers_to_freeze = 172
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[num_layers_to_freeze:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    history_fine_tune = model.fit_generator(
        train_gen,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=num_epoch,
        validation_data=valid_gen,
        validation_steps=num_validate_samples // batch_size,
        class_weight='auto')

    # Save fine tuned model
    model.save('inceptionv3-fine-tune.model')