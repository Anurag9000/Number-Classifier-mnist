import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def test_gpu():
    if tf.config.list_physical_devices('GPU'):
        print("✅ TensorFlow is using the GPU!")
    else:
        print("❌ No GPU detected. Using CPU.")
# ======================
# Data Loading & Preprocessing
# ======================
class DataLoader:
    def __init__(self, batch_size, strategy, dataset_name="mnist"):
        self.batch_size = batch_size
        self.strategy = strategy
        self.dataset_name = dataset_name

    @staticmethod
    def read_image_tfds(image, label):
        # Use random offset padding: generate random top-left offset in [0, 48]
        xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
        ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
        image = tf.reshape(image, [28, 28, 1])
        image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
        image = tf.cast(image, tf.float32) / 255.0

        xmax = xmin + 28
        ymax = ymin + 28

        xmin = tf.cast(xmin, tf.float32)
        ymin = tf.cast(ymin, tf.float32)
        xmax = tf.cast(xmax, tf.float32)
        ymax = tf.cast(ymax, tf.float32)
        return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])

    def get_training_dataset(self):
        with self.strategy.scope():
            dataset = tfds.load(self.dataset_name, split="train", as_supervised=True, try_gcs=True)
            dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
            dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_validation_dataset(self):
        with self.strategy.scope():
            dataset = tfds.load(self.dataset_name, split="train", as_supervised=True, try_gcs=True)
            dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
            dataset = dataset.batch(10000, drop_remainder=True)
        return dataset

    def get_test_dataset(self):
        with self.strategy.scope():
            dataset = tfds.load(self.dataset_name, split="test", as_supervised=True, try_gcs=True)
            dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
            dataset = dataset.batch(10000, drop_remainder=True)
        return dataset

    @staticmethod
    def dataset_to_numpy_util(training_dataset, validation_dataset, N):
        # Unbatch and then batch N samples from each dataset for inspection
        train_ds = training_dataset.unbatch().batch(N)
        val_ds = validation_dataset.unbatch().batch(N)
        training_digits, training_labels, training_bboxes = None, None, None
        validation_digits, validation_labels, validation_bboxes = None, None, None
        if tf.executing_eagerly():
            for digits, (labels, bboxes) in train_ds:
                training_digits = digits.numpy()
                training_labels = np.argmax(labels.numpy(), axis=1)
                training_bboxes = bboxes.numpy()
                break
            for digits, (labels, bboxes) in val_ds:
                validation_digits = digits.numpy()
                validation_labels = np.argmax(labels.numpy(), axis=1)
                validation_bboxes = bboxes.numpy()
                break
        return (training_digits, training_labels, training_bboxes,
                validation_digits, validation_labels, validation_bboxes)

# ======================
# Visualization Utilities
# ======================
class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=2, display_str_list=None, use_normalized_coordinates=True):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            left, right = xmin * im_width, xmax * im_width
            top, bottom = ymin * im_height, ymax * im_height
        else:
            left, right, top, bottom = xmin, xmax, ymin, ymax
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        if display_str_list:
            text_position = (left + 5, top - 10)
            for display_str in display_str_list:
                draw.text(text_position, display_str, fill=color)
                text_position = (text_position[0], text_position[1] + 15)

    @staticmethod
    def draw_bounding_boxes_on_image(image, boxes, colors=None, thickness=2, display_str_list=None):
        boxes_shape = boxes.shape
        if len(boxes_shape) != 2 or boxes_shape[1] != 4:
            raise ValueError("Boxes should be a 2D array of shape (N, 4)")
        if colors is None:
            colors = ['red'] * boxes.shape[0]
        if display_str_list is None:
            display_str_list = [None] * boxes.shape[0]
        for i in range(boxes.shape[0]):
            Visualizer.draw_bounding_box_on_image(
                image,
                ymin=boxes[i][0],
                xmin=boxes[i][1],
                ymax=boxes[i][2],
                xmax=boxes[i][3],
                color=colors[i],
                thickness=thickness,
                display_str_list=[display_str_list[i]] if display_str_list[i] else None
            )

    @staticmethod
    def draw_bounding_boxes_on_image_array(image, boxes, colors=None, display_str_list=None):
        # image: numpy array (H, W) or (H, W, 1)
        if image.ndim == 2:
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
        elif image.ndim == 3 and image.shape[-1] == 1:
            pil_image = Image.fromarray(image.squeeze().astype(np.uint8), mode='L')
        else:
            raise ValueError("Unsupported image shape: {}".format(image.shape))
        Visualizer.draw_bounding_boxes_on_image(pil_image, boxes, colors, thickness=2, display_str_list=display_str_list)
        return np.array(pil_image)

    @staticmethod
    def display_digits_with_bboxes(digits, predictions, labels, pred_bboxes, true_bboxes, iou, title):
        # Display 10 images: 5 on top row, 5 on bottom row
        n = 10
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(title, fontsize=16)
        indexes = np.random.choice(len(predictions), size=n, replace=False)
        for i, ax in enumerate(axes.flatten()):
            idx = indexes[i]
            img = (digits[idx] * 255).astype(np.uint8).reshape(75, 75)
            pil_img = Image.fromarray(img).convert("RGB")
            # Draw two boxes: true (green) and predicted (red)
            boxes_to_draw = np.array([true_bboxes[idx], pred_bboxes[idx]])
            colors = ['green', 'red']
            display_str_list = ['True', 'Pred']
            drawn_img = Visualizer.draw_bounding_boxes_on_image_array(img, boxes_to_draw, colors, display_str_list)
            ax.imshow(drawn_img)
            ax.axis('off')
            ax.set_title(f"Pred: {predictions[idx]}, IoU: {iou[idx]:.2f}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history.history['class_output_accuracy'], label='Train Acc')
        if 'val_class_output_accuracy' in history.history:
            plt.plot(history.history['val_class_output_accuracy'], label='Val Acc')
        plt.title('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(history.history['class_output_loss'], label='Train Loss')
        if 'val_class_output_loss' in history.history:
            plt.plot(history.history['val_class_output_loss'], label='Val Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(history.history['bounding_box_loss'], label='Train BBox MSE')
        if 'val_bounding_box_loss' in history.history:
            plt.plot(history.history['val_bounding_box_loss'], label='Val BBox MSE')
        plt.title('BBox Regression Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        plt.show()

# ======================
# Model Building
# ======================
class ModelBuilder:
    @staticmethod
    def feature_extractor(inputs):
        x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(inputs)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=3)(x)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=3)(x)
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)
        return x

    @staticmethod
    def dense_layers(inputs):
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        return x

    @staticmethod
    def classifier(inputs):
        classification_output = tf.keras.layers.Dense(10, activation='softmax', name="class_output")(inputs)
        return classification_output

    @staticmethod
    def bounding_box_regression(inputs):
        bounding_box_output = tf.keras.layers.Dense(4, name="bounding_box")(inputs)
        return bounding_box_output

    @classmethod
    def build_model(cls, inputs):
        feature_cnn = cls.feature_extractor(inputs)
        dense_output = cls.dense_layers(feature_cnn)
        classification_output = cls.classifier(dense_output)
        bounding_box_output = cls.bounding_box_regression(dense_output)
        model = tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])
        return model

    @classmethod
    def compile_model(cls, inputs):
        model = cls.build_model(inputs)
        model.compile(
            optimizer='adam',
            loss={
                'class_output': tf.keras.losses.CategoricalCrossentropy(),
                'bounding_box': tf.keras.losses.MeanSquaredError()
            },
            metrics={
                'class_output': 'accuracy',
                'bounding_box': 'mse'
            }
        )
        return model

# ======================
# Metrics
# ======================
class Metrics:
    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        return interArea / (boxAArea + boxBArea - interArea + 1e-5)

# ======================
# Object Detector Pipeline
# ======================
class ObjectDetector:
    def __init__(self, strategy, batch_size, save_path="D:/models", dataset_name="mnist"):
        self.strategy = strategy
        self.batch_size = batch_size
        self.save_path = save_path
        self.data_loader = DataLoader(batch_size, strategy, dataset_name)
        self.visualizer = Visualizer()
        self.model = None
        self.checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, "best_model.h5"),
            monitor="val_class_output_accuracy",
            save_best_only=True,
            save_weights_only=False
        )

    def build_and_compile_model(self):
        inputs = tf.keras.Input(shape=(75, 75, 1))
        self.model = ModelBuilder.compile_model(inputs)
        return self.model

    def train(self, training_dataset, validation_dataset, epochs, steps_per_epoch):
        self.model.fit(
            training_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=1,
            epochs=epochs,
            callbacks=[self.checkpoint_cb]
        )

    def predict(self, digits):
        return self.model.predict(digits)

    def save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save(os.path.join(self.save_path, "object_detector.keras"))
        self.model.save_weights(os.path.join(self.save_path, "object_detector.weights.h5"))

    def load_model(self):
        model_path = os.path.join(self.save_path, "object_detector.keras")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            inputs = tf.keras.Input(shape=(75, 75, 1))
            self.model = ModelBuilder.compile_model(inputs)
            history = self.model.fit(
                self.data_loader.get_training_dataset(),
                steps_per_epoch=60000 // self.batch_size,
                validation_data=self.data_loader.get_validation_dataset(),
                validation_steps=1,
                epochs=20,
                callbacks=[self.checkpoint_cb]
            )
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(model_path)

# ======================
# Main Pipeline
# ======================
def main():
    test_gpu()
    strategy = tf.distribute.get_strategy()
    batch_size = 32 * strategy.num_replicas_in_sync
    obj_detector = ObjectDetector(strategy, batch_size)

    # Get datasets
    training_dataset = obj_detector.data_loader.get_training_dataset()
    validation_dataset = obj_detector.data_loader.get_validation_dataset()

    # Extract a few samples for inspection
    training_digits, training_labels, training_bboxes, \
    validation_digits, validation_labels, validation_bboxes = DataLoader.dataset_to_numpy_util(
        training_dataset, validation_dataset, 10
    )

    # Build, compile, and train the model
    obj_detector.build_and_compile_model()
    obj_detector.train(training_dataset, validation_dataset, epochs=40, steps_per_epoch=60000 // batch_size)
    obj_detector.save_model()

    # Predictions on validation set
    pred_probs, pred_bboxes = obj_detector.model.predict(validation_digits)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Compute IoU scores for each sample
    iou_scores = []
    for pred_box, true_box in zip(pred_bboxes, validation_bboxes):
        iou_scores.append(Metrics.compute_iou(pred_box, true_box))
    iou_scores = np.array(iou_scores)

    obj_detector.visualizer.display_digits_with_bboxes(
        validation_digits,
        pred_labels,
        validation_labels,
        pred_bboxes,
        validation_bboxes,
        iou_scores,
        "Validation Predictions"
    )

    # Optionally plot training history if available
    # Visualizer.plot_training_history(history)

if __name__ == "__main__":
    main()
