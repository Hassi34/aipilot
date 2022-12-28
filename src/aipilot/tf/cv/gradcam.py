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
            # First, we create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions
            grad_model = tf.keras.models.Model(
                [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
            )

            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, self.preds = grad_model(self.array)
                if pred_index is None:
                    pred_index = tf.argmax(self.preds[0])
                class_channel = self.preds[:, pred_index]

            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)

            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            self.heatmap = heatmap.numpy()
        except AttributeError:
            self._get_img_array()
            self.make_gradcam_heatmap()
        return self.heatmap, self.preds.numpy()

    def save_and_display_gradcam(self, alpha=0.4):
        try:
            # Load the original image
            img = tf.keras.preprocessing.image.load_img(self.in_img_path)
            img = tf.keras.preprocessing.image.img_to_array(img)

            # Rescale heatmap to a range 0-255
            self.heatmap = np.uint8(255 * self.heatmap)

            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[self.heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * alpha + img
            superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

            # Save the superimposed image
            superimposed_img.save(self.out_img_path)

            # Display Grad CAM
            display(Image(self.out_img_path))
        except AttributeError:
            self.make_gradcam_heatmap()
            self.save_and_display_gradcam()