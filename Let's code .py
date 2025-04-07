import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw
import PIL
import matplotlib.pyplot as plt

save_path = "D:/models"
def draw_bounding_boxes_on_image_array(image, boxes, color=None, display_str_list=None):
    """
    Args:
        image: 2D or 3D NumPy array, shape (H, W) or (H, W, 1)
        boxes: np.array of shape (N, 4) — ymin, xmin, ymax, xmax
        color: List of colors (optional)
        display_str_list: List of labels for each box (optional)
    Returns:
        np.array: image with boxes drawn (same shape as input)
    """
    if image.ndim == 2:
        pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
    elif image.ndim == 3 and image.shape[-1] == 1:
        pil_image = Image.fromarray(image.squeeze().astype(np.uint8), mode='L')
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))

    draw_bounding_boxes_on_image(
        pil_image, boxes, color=color, display_str_list=display_str_list
    )

    return np.array(pil_image)

def get_test_dataset():
    with strategy.scope():
        dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.batch(10000, drop_remainder=True)
    return dataset

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                                color='red', thickness=1,
                                display_str_list=None,
                                use_normalized_coordinates=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        left, right = xmin * im_width, xmax * im_width
        top, bottom = ymin * im_height, ymax * im_height
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)],
              width=thickness, fill=color)

    if display_str_list:
        text_position = (left + 5, top - 10)
        for display_str in display_str_list:
            draw.text(text_position, display_str, fill=color)
            text_position = (text_position[0], text_position[1] + 15)

def draw_bounding_boxes_on_image(image, boxes, color=None, thickness=1, display_str_list=None):
    boxes_shape = boxes.shape
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Boxes should be a 2D array of shape (N, 4)")

    if color is None:
        color = ['red'] * boxes.shape[0]
    if display_str_list is None:
        display_str_list = [None] * boxes.shape[0]

    for i in range(boxes.shape[0]):
        draw_bounding_box_on_image(
            image,
            ymin=boxes[i][0],
            xmin=boxes[i][1],
            ymax=boxes[i][2],
            xmax=boxes[i][3],
            color=color[i],
            thickness=thickness,
            display_str_list=[display_str_list[i]] if display_str_list[i] else None
        )

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
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

MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

def dataset_to_numpy(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for digits, (labels, bboxes) in dataset.take(1):
        digits = digits.numpy()
        labels = np.argmax(labels.numpy(), axis=1)
        bboxes = bboxes.numpy()
        return digits, labels, bboxes

def create_digits_from_local_fonts(n):
    font_labels = []
    img = PIL.Image.new('L', (75 * n, 75), color=(0, 255))
    font1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono-Oblique.ttf'), 25)
    font2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 25)
    d = PIL.ImageDraw.Draw(img)

    for i in range(n):
        font_labels.append(i % 10)
        d.text((75 * i, 0), str(i % 10), fill=(255, 255), font=font1 if i < 10 else font2)

    font_digits = np.array(img.getdata()).reshape((75, 75 * n)) / 255.0
    font_digits = np.reshape(font_digits, (n, 75, 75, 1))
    
    return font_digits, np.array(font_labels)

def display_digits_with_bboxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 3
    indexes = np.random.choice(len(predictions), size=n)
    
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]
    n_iou = []

    if len(iou) > 0:
        n_iou = iou[indexes]

    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes]

    if len(bboxes) > 0:
        n_bboxes = bboxes[indexes]

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(-1, 75, 75)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        bboxes_to_plot = []

        if len(pred_bboxes) > i:
            bboxes_to_plot.append(n_pred_bboxes[i])

        if len(bboxes) > i:
            bboxes_to_plot.append(n_bboxes[i])

        img_to_draw = draw_bounding_boxes_on_image_array(
            image=n_digits[i],
            boxes=np.asarray(bboxes_to_plot),
            color=['red', 'green'],
            display_str_list=['True', 'Pred']
        )

        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')

        plt.imshow(img_to_draw)

        if len(iou) > i:
            if i < len(n_iou) and n_iou[i] < 0.5:
                color = 'red'


strategy = tf.distribute.get_strategy()
strategy.num_replicas_in_sync

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

def read_image_tfds(image, label):
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

def get_training_dataset():
    with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(-1)
    return dataset

def get_validation_dataset():
    with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.batch(10000, drop_remainder=True)
    return dataset

with strategy.scope():
    training_dataset = get_training_dataset()
    validation_dataset = get_validation_dataset()

training_digits, training_labels, training_bboxes, \
validation_digits, validation_labels, validation_bboxes = dataset_to_numpy_util(training_dataset, validation_dataset, 10)

display_digits_with_bboxes(
    validation_digits,
    validation_labels,
    validation_labels,
    np.zeros((10, 4)),
    validation_bboxes,
    np.ones(10),
    "Validation Digits & Labels"
)

# Cell [19]
def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    return x

# Cell [20]
def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x

# Cell [21]
def classifier(inputs):
    classification_output = tf.keras.layers.Dense(10, activation='softmax', name="class_output")(inputs)
    return classification_output

# Cell [22]
def bounding_box_regression(inputs):
    bounding_box_output = tf.keras.layers.Dense(4, name="bounding_box")(inputs)
    return bounding_box_output

def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)

    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])

    return model

def define_and_compile_model(inputs):
    model = final_model(inputs)

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

# 5: Train and Validate the Model
EPOCHS = 40
steps_per_epoch = 60000 // BATCH_SIZE

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.h5",
    monitor="val_class_output_accuracy",
    save_best_only=True,
    save_weights_only=False
)

inputs = tf.keras.Input(shape=(75, 75, 1))
model = define_and_compile_model(inputs) 

history = model.fit(
    training_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=1,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-5)
    return iou

# Get predictions
pred_probs, pred_bboxes = model.predict(validation_digits)

# Predicted labels from softmax
pred_labels = np.argmax(pred_probs, axis=1)

iou_scores = []
for pred_box, true_box in zip(pred_bboxes, validation_bboxes):
    iou_scores.append(compute_iou(pred_box, true_box))
iou_scores = np.array(iou_scores)

display_digits_with_bboxes(
    validation_digits,
    validation_labels,
    validation_labels,
    np.zeros((10, 4)),
    validation_bboxes,
    np.ones(10),
    "Validation Digits & Labels"
)


def plot_training_history(history):
    # Plot classification accuracy
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['class_output_accuracy'], label='Train Acc')
    if 'val_class_output_accuracy' in history.history:
        plt.plot(history.history['val_class_output_accuracy'], label='Val Acc')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot classification loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['class_output_loss'], label='Train Loss')
    if 'val_class_output_loss' in history.history:
        plt.plot(history.history['val_class_output_loss'], label='Val Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot bounding box regression loss (MSE)
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



os.makedirs(save_path, exist_ok=True)
model.save(os.path.join(save_path, "object_detector.keras"))

# Save and load weights
model.save_weights(os.path.join(save_path, "object_detector.weights.h5"))

inputs = tf.keras.Input(shape=(75, 75, 1))
model = define_and_compile_model(inputs)
model.load_weights("D:/models/object_detector.weights.h5")


if os.path.exists("D:/models/object_detector.keras"):
    model = tf.keras.models.load_model("D:/models/object_detector.keras")
else:
    inputs = tf.keras.Input(shape=(75, 75, 1))
    model = define_and_compile_model(inputs)
    history = model.fit(
        training_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=1,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb]
    )
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "object_detector.keras"))


test_dataset = get_test_dataset()

# Convert to NumPy
test_digits, test_labels, test_bboxes = dataset_to_numpy(validation_dataset=test_dataset, N=10)

pred_probs, pred_bboxes = model.predict(test_digits)
pred_labels = np.argmax(pred_probs, axis=1)

# IoU
iou_scores = []
for pred_box, true_box in zip(pred_bboxes, test_bboxes):
    iou_scores.append(compute_iou(pred_box, true_box))
iou_scores = np.array(iou_scores)

display_digits_with_bboxes(
    validation_digits,
    validation_labels,
    validation_labels,
    np.zeros((10, 4)),
    validation_bboxes,
    np.ones(10),
    "Validation Digits & Labels"
)


