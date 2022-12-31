import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class DataPrep:
    def __init__(self, data_dir):
        self.data_dir  = data_dir
    def data_generators(self, val_split = 0.2, img_size = (224, 224), batch_size = 16,
                            data_augmentation = False, train_dir = None, val_dir = None,
                            train_shuffle = True, val_shuffle = False):
        if val_dir is not None:
            datagen_kwargs = dict( rescale=1./255)
        else:
            datagen_kwargs = dict( rescale=1./255, validation_split= val_split )

        dataflow_kwargs = dict(
                target_size=img_size,
                batch_size = batch_size,
                interpolation="bilinear")

        base_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
            
        if data_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                shear_range=0.2,
                rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # Randomly zoom image 
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,
                **datagen_kwargs
            )
        else:
            train_datagen = base_datagen
        if train_dir is not None:
            train_generator = train_datagen.flow_from_directory(
                directory=train_dir,
                subset="training",
                shuffle=train_shuffle,
                **dataflow_kwargs)
        else:
            train_generator = train_datagen.flow_from_directory(
                directory=self.data_dir,
                subset="training",
                shuffle=train_shuffle,
                **dataflow_kwargs)
            
        if val_dir is not None:
                valid_generator = base_datagen.flow_from_directory(
                directory=val_dir,
                shuffle=val_shuffle,
                **dataflow_kwargs)
        else:
            try:
                valid_generator = base_datagen.flow_from_directory(
                    directory=train_dir,
                    subset="validation",
                    shuffle=val_shuffle,
                    **dataflow_kwargs)
            except TypeError:
                valid_generator = base_datagen.flow_from_directory(
                    directory=self.data_dir,
                    subset="validation",
                    shuffle=val_shuffle,
                    **dataflow_kwargs)

        return train_generator, valid_generator

    def classwise_img_count(self, generator: object) -> dict:
        class_names = list(generator.class_indices.keys())
        labels = generator.labels
        data = dict.fromkeys(class_names, 0)
        for label in labels:
            label = class_names[label]
            data[label] += 1
        return data
    
    def sample_images(self, generator: object):
        classes = list(generator.class_indices.keys())
        plt.figure(figsize=(10, 10))
        images, labels = generator.next()
        labels = np.argmax(labels, axis=1)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(classes[labels[i]])
            plt.axis("off")