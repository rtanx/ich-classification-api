import numpy as np
import tensorflow as tf
import cv2
from types import MappingProxyType


class GradCAM:
    def __init__(self, model, last_conv_layer_name):
        """
        Initialize the GradCAM object.

        Args:
        - model: Trained TensorFlow/Keras model.
        - last_conv_layer_name: Name of the last convolutional layer in the model.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name

        # Create a model that outputs the activations of the last conv layer and predictions
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )

    def compute_heatmap(self, image, class_idx):
        """
        Compute Grad-CAM heatmap for a specific class index.

        Args:
        - image: Preprocessed input image of shape (H, W, C).
        - class_idx: Index of the class to compute the Grad-CAM for.

        Returns:
        - heatmap: Grad-CAM heatmap for the given class index.
        """
        # Add a batch dimension to the image
        image = np.expand_dims(image, axis=0)

        # Record the gradients of the target class output w.r.t. the last conv layer
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(image)
            target_class_output = predictions[:, class_idx]

        # Compute the gradients of the target class output with respect to conv_output
        grads = tape.gradient(target_class_output, conv_output)

        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the convolutional layer output with the pooled gradients
        conv_output = conv_output[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

        # Normalize the heatmap to range [0, 1]
        heatmap = np.maximum(heatmap, 0)  # ReLU activation to keep positive values only
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Avoid division by zero
        return heatmap

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
        - heatmap: The heatmap to overlay (2D array).
        - original_image: Original image before preprocessing (H, W, C).
        - alpha: Transparency factor for overlaying the heatmap.
        - colormap: OpenCV colormap to apply to the heatmap.

        Returns:
        - overlayed_image: Image with the heatmap overlayed.
        """
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

        # Convert original_image to uint8 if it's not already
        if original_image.dtype != np.uint8:
            original_image = np.uint8(255 * original_image)

        # Blend the heatmap with the original image
        overlayed_image = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlayed_image

    def generate_heatmaps(self, image, original_image, class_indices, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Generate Grad-CAM heatmaps for multiple class indices.

        Args:
        - image: Preprocessed input image of shape (H, W, C).
        - original_image: Original image before preprocessing (used for overlaying).
        - class_indices: List of class indices to compute heatmaps for.
        - alpha: Transparency factor for overlaying the heatmap.
        - colormap: OpenCV colormap to apply to the heatmap.

        Returns:
        - heatmaps: Dictionary where keys are class indices and values are overlayed images.
        """
        heatmaps = {}
        for class_idx in class_indices:
            heatmap = self.compute_heatmap(image, class_idx)
            overlayed_image = self.overlay_heatmap(heatmap, original_image, alpha, colormap)
            heatmaps[class_idx] = overlayed_image
        return heatmaps


ich_subtype_index_labels = MappingProxyType(
    {
        0: 'any',
        1: 'epidural',
        2: 'intraparenchymal',
        3: 'intraventricular',
        4: 'subarachnoid',
        5: 'subdural',
    }
)


def decode_hot_encoded_labels(y_preds: np.ndarray, threshold: float = 0.5):
    result = []
    for y_pred in y_preds:
        selected_indices = np.argwhere(y_pred > threshold).squeeze()
        result.append([ich_subtype_index_labels[idx] for idx in selected_indices])

    return np.array(result, dtype=str)


def hot_encoded_indices_to_labels(indices: np.ndarray):
    result = []
    for idx in indices:
        result.append(ich_subtype_index_labels[idx])

    return np.array(result, dtype=str)
