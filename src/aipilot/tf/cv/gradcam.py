import tensorflow as tf
import matplotlib.cm as cm 
from IPython.display import Image, display
import numpy as np

class GradCam:
    def __init__(self, model, last_conv_layer_name, in_img_path, out_img_path = "out.png", img_size = (224,224)):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.out_img_path = out_img_path
        self.in_img_path = in_img_path
        self.img_size = img_size

    def _get_img_array(self):
        img = tf.keras.preprocessing.image.load_img(self.in_img_path, target_size=self.img_size)
        self.array = tf.keras.preprocessing.image.img_to_array(img)
        self.array = np.expand_dims(self.array, axis=0)
        return self.array

    def make_gradcam_heatmap(self, pred_index=None):
        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
            )
            with tf.GradientTape() as tape:
                last_conv_layer_output, self.preds = grad_model(self.array)
                if pred_index is None:
                    pred_index = tf.argmax(self.preds[0])
                class_channel = self.preds[:, pred_index]

            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            self.heatmap = heatmap.numpy()
        except AttributeError:
            self._get_img_array()
            self.make_gradcam_heatmap()
        return self.heatmap, self.preds.numpy()

    def get_gradcam(self, alpha=0.4):
        try:
            img = tf.keras.preprocessing.image.load_img(self.in_img_path)
            img = tf.keras.preprocessing.image.img_to_array(img)

            self.heatmap = np.uint8(255 * self.heatmap)

            jet = cm.get_cmap("jet")

            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[self.heatmap]
            
            jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

            superimposed_img = jet_heatmap * alpha + img
            superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

            superimposed_img.save(self.out_img_path)

            display(Image(self.out_img_path))
            
        except AttributeError:
            self.make_gradcam_heatmap()
            self.get_gradcam()

    def display(self):
        try:
            display(Image(self.out_img_path))
        except FileNotFoundError:
            self.get_gradcam()
            display()