import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ==== Load Model ====
model = tf.keras.models.load_model("D:/models/object_detector.keras")
print("✅ Model loaded successfully.")

# ==== Preprocess Function ====
def read_image_tfds(image, label):
    # Reshape & normalize
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32) / 255.0

    # Get bounding box of non-zero pixels
    non_zero = tf.where(image[..., 0] > 0.05)
    ymin = tf.cast(tf.reduce_min(non_zero[:, 0]), tf.int32)
    ymax = tf.cast(tf.reduce_max(non_zero[:, 0]), tf.int32)
    xmin = tf.cast(tf.reduce_min(non_zero[:, 1]), tf.int32)
    xmax = tf.cast(tf.reduce_max(non_zero[:, 1]), tf.int32)

    # For simplicity, we pad the 28x28 image to 75x75 with a random offset.
    # Maximum offset = 75 - 28 = 47.
    pad_top = tf.random.uniform((), 0, 47 + 1, dtype=tf.int32)
    pad_left = tf.random.uniform((), 0, 47 + 1, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, 75, 75)

    # Update bounding box coordinates with the padding offset.
    xmin_new = tf.cast(pad_left + xmin, tf.float32)
    ymin_new = tf.cast(pad_top + ymin, tf.float32)
    xmax_new = tf.cast(pad_left + xmax, tf.float32)
    ymax_new = tf.cast(pad_top + ymax, tf.float32)
    bbox = [ymin_new, xmin_new, ymax_new, xmax_new]
    
    return image, (tf.one_hot(label, 10), bbox)

# ==== Load Dataset ====
def get_test_dataset():
    dataset = tfds.load("mnist", split="test", as_supervised=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(10, drop_remainder=True)
    return dataset

# ==== IoU Calculation ====
def compute_iou(boxA, boxB):
    # Boxes are in format: [ymin, xmin, ymax, xmax]
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0])
    boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
    return interArea / (boxAArea + boxBArea - interArea + 1e-5)

# ==== Visualization ====
def display_digits_with_bboxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 10
    rows = 2
    cols = 5
    # Increase figure size to show bigger images & boxes.
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    indexes = np.random.choice(len(predictions), size=n, replace=False)
    
    for i, ax in enumerate(axes.flatten()):
        idx = indexes[i]
        img = (digits[idx] * 255).astype(np.uint8).reshape(75, 75)
        pil_img = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        
        # Get GT (green) & Predicted (red) bounding boxes.
        gt_box = bboxes[idx]      # [ymin, xmin, ymax, xmax]
        pred_box = pred_bboxes[idx]  # [ymin, xmin, ymax, xmax]
        
        # Draw boxes with thicker lines
        draw.rectangle([(gt_box[1], gt_box[0]), (gt_box[3], gt_box[2])], outline="green", width=3)
        draw.rectangle([(pred_box[1], pred_box[0]), (pred_box[3], pred_box[2])], outline="red", width=3)
        
        # Display labels on image
        draw.text((4, 4), f"True: {labels[idx]}", fill="green")
        draw.text((pred_box[1], pred_box[2]), f"Pred: {predictions[idx]}", fill="red")
        
        ax.imshow(pil_img)
        ax.axis('off')
        ax.set_title(f"IoU: {iou[idx]:.2f}", fontsize=10, color='green' if iou[idx] > 0.5 else 'red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

# ==== Run Prediction ====
test_dataset = get_test_dataset()

for test_digits, (test_labels_onehot, test_bboxes) in test_dataset.take(1):
    test_digits = test_digits.numpy()
    test_labels = np.argmax(test_labels_onehot.numpy(), axis=1)
    test_bboxes = test_bboxes.numpy()

    pred_probs, pred_bboxes = model.predict(test_digits)
    pred_labels = np.argmax(pred_probs, axis=1)

    ious = np.array([compute_iou(pred, true) for pred, true in zip(pred_bboxes, test_bboxes)])

    display_digits_with_bboxes(
        test_digits, pred_labels, test_labels, pred_bboxes, test_bboxes, ious,
        "🔍 Object Detector: Predictions vs Ground Truth"
    )
