# standard imports
from matplotlib import pyplot as plt
from skimage.io import imread
import numpy as np
import datetime
import math
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence
import zipfile
import os

data_zip_path = "data/lunar_dataset.zip"
extract_dir = "data/lunar_dataset"

if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
else:
    print("Dataset already extracted.")

# Optional: ensure segmentation_models is installed
try:
    import segmentation_models as sm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "segmentation_models"])
    import segmentation_models as sm

# Set framework for segmentation_models
os.environ["SM_FRAMEWORK"] = "tf.keras"
sm.set_framework('tf.keras')

# Set image data format explicitly
tf.keras.backend.set_image_data_format('channels_last')

# Define dataset class
class LunarDataset(Sequence):
    def __init__(self, x_set, y_set, batch_size, dims, classes):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.img_height, self.img_width = dims
        self.classes = classes

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = (idx + 1) * self.batch_size
        batch_x = self.x[start_index:end_index]
        batch_y = self.y[start_index:end_index]

        xtr, ytr = [], []
        for filename_x, filename_y in zip(batch_x, batch_y):
            img = imread(filename_x)[:self.img_height, :self.img_width, :] / 255.0
            img = img.astype(np.float32)
            xtr.append(img)

            mask = imread(filename_y, as_gray=True)[:self.img_height, :self.img_width] // 0.07
            mask[mask == 3] = 2
            mask[mask == 10] = 3
            mask = to_categorical(mask, num_classes=self.classes)
            ytr.append(mask)

        return np.array(xtr), np.array(ytr).astype(np.float32)

# Load dataset
img_dir = 'data/lunar_dataset/images/render'
mask_dir = 'data/lunar_dataset/images/clean'


images = [os.path.join(img_dir, x) for x in sorted(os.listdir(img_dir))]
masks = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir))]

X_train = images[:8000]
y_train = masks[:8000]
X_valid = images[8000:-4]
y_valid = masks[8000:-4]
X_test = images[-4:]
y_test = masks[-4:]

# Create dataset
batch_size = 16
dims = (480, 480)
num_classes = 4

train_dataset = LunarDataset(X_train, y_train, batch_size, dims, num_classes)
valid_dataset = LunarDataset(X_valid, y_valid, batch_size, dims, num_classes)

# Visualize one batch
batch = next(iter(train_dataset))
sample = batch[1][1]

fig, ((a1, a2, a3), (a4, a5, a6)) = plt.subplots(2, 3, figsize=(10, 8))
for i, (ax, title) in enumerate(zip((a1, a2, a3, a4, a5, a6),
                                    ('Original', 'Combined Mask', 'Background', 'Large Rocks', 'Sky', 'Small Rocks'))):
    if i == 0:
        ax.imshow(batch[0][1])
    elif i == 1:
        ax.imshow(np.argmax(sample, axis=-1))
    else:
        ax.imshow(sample[:, :, i - 2])
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Model setup
BACKBONE = 'vgg16'
input_shape = (480, 480, 3)
activation = 'softmax'

model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=input_shape,
    classes=num_classes,
    activation=activation,
    encoder_weights='imagenet',
    encoder_freeze=True
)

model.summary()

# Compile model
lr = 1e-3
epochs = 5
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr),
    metrics=metrics
)

train_steps = len(X_train) // batch_size
valid_steps = len(X_valid) // batch_size
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/lunarModel_{current_datetime}.h5',
        monitor='val_iou_score',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_iou_score",
        mode='max',
        patience=2,
        factor=0.1,
        verbose=1,
        min_lr=1e-6
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_score",
        patience=3,
        verbose=1,
        mode='max'
    )
]

# Train the model
model_history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    validation_data=valid_dataset,
    validation_steps=valid_steps,
    epochs=epochs,
    callbacks=callbacks
)

# Prediction helper
def predict_image(img_path, mask_path, model):
    img = imread(img_path)[:480, :480, :] / 255.0
    img = img.astype(np.float32)

    mask = imread(mask_path, as_gray=True)[:480, :480]
    pred_mask = model.predict(np.expand_dims(img, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)[0]

    inter = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    iou = inter.sum() / union.sum()

    return img, mask, pred_mask, iou

# Test prediction
img_path = X_test[0]
mask_path = y_test[0]
img, mask, pred_mask, iou = predict_image(img_path, mask_path, model)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
ax1.set_title("Input Image")
ax1.imshow(img)
ax2.set_title("True Mask")
ax2.imshow(mask)
ax3.set_title(f"Predicted Mask (IOU: {iou:.2f})")
ax3.imshow(pred_mask)
plt.show()
