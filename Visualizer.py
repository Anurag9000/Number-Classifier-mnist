import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

##################################################
# 1. Preprocessing: Same offset logic
##################################################
def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0

    xmax = xmin + 28
    ymax = ymin + 28

    # Convert to float for bounding box
    bbox = [tf.cast(val, tf.float32) for val in [ymin, xmin, ymax, xmax]]
    label_onehot = tf.one_hot(label, 10)
    return image, (label_onehot, bbox)

##################################################
# 2. Load a small batch of "validation" data
##################################################
def get_validation_data(num_samples=10):
    # Using 'train' split as validation here (demonstration).
    dataset = tfds.load("mnist", split="train", as_supervised=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(num_samples).take(1)

    for digits, (labels, bboxes) in dataset:
        digits = digits.numpy()
        labels = np.argmax(labels.numpy(), axis=1)  # convert one-hot to class indices
        bboxes = bboxes.numpy()
        return digits, labels, bboxes

##################################################
# 3. Compute IoU
##################################################
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-5)

##################################################
# 4. Display bounding boxes (like your Visualizer code)
##################################################
def draw_bounding_boxes_on_image_array(image, boxes, colors=None, labels=None):
    """
    image: NumPy array of shape (75,75) or (75,75,1)
    boxes: shape (N,4) => [ymin, xmin, ymax, xmax]
    colors: list of color strings
    labels: list of label strings
    """
    if image.ndim == 2:
        pil_img = Image.fromarray(image.astype(np.uint8), 'L').convert("RGB")
    else:
        pil_img = Image.fromarray(image.squeeze().astype(np.uint8), 'L').convert("RGB")

    draw = ImageDraw.Draw(pil_img)
    if colors is None:
        colors = ['red'] * len(boxes)
    if labels is None:
        labels = [None] * len(boxes)

    for box, color, label in zip(boxes, colors, labels):
        ymin, xmin, ymax, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        if label:
            text_pos = (xmin + 2, ymax + 2)
            draw.text(text_pos, label, fill=color)
    return np.array(pil_img)

def display_digits_with_bboxes(digits, predictions, true_labels, pred_bboxes, true_bboxes, ious, title):
    n_show = 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)

    # Randomly sample indices
    if len(predictions) < n_show:
        n_show = len(predictions)
    idxs = np.random.choice(len(predictions), size=n_show, replace=False)

    for i, ax in enumerate(axes.flatten()):
        if i >= n_show:
            break
        idx = idxs[i]

        # Convert digit to display
        img = (digits[idx] * 255).astype(np.uint8).reshape(75, 75)

        # 2 boxes: green = true, red = pred
        boxes = np.array([true_bboxes[idx], pred_bboxes[idx]])
        colors = ['green', 'red']
        labels = [f"True: {true_labels[idx]}", f"Pred: {predictions[idx]}"]

        # Draw
        annotated_img = draw_bounding_boxes_on_image_array(img, boxes, colors, labels)
        ax.imshow(annotated_img)
        ax.axis('off')
        ax.set_title(f"IoU: {ious[idx]:.2f}", color=('green' if ious[idx] >= 0.5 else 'red'))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

##################################################
# 5. Main: Load model, get data, predict, visualize
##################################################
def main():
    # Load your saved model
    model_path = "D:/models/object_detector.keras"
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Get a small batch of data
    digits, true_labels, true_bboxes = get_validation_data(num_samples=20)

    # Predict
    pred_probs, pred_bboxes = model.predict(digits)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Compute IoU for each sample
    ious = [compute_iou(pb, tb) for pb, tb in zip(pred_bboxes, true_bboxes)]
    ious = np.array(ious)

    # Show 10 random images
    display_digits_with_bboxes(
        digits,
        pred_labels,
        true_labels,
        pred_bboxes,
        true_bboxes,
        ious,
        title="Validation Predictions"
    )

if __name__ == "__main__":
    main()
