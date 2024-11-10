# Lunar Crater Detection

## Project Overview
This project involves detecting lunar craters from images using deep learning. The model is designed to perform **segmentation** on satellite images to identify craters on the moon's surface. The dataset consists of labeled images, where each image has a corresponding label file in YOLO format.

The model is built using **U-Net architecture** for segmentation, with the goal of detecting and segmenting lunar craters from satellite images.

## Features
- **Image Segmentation**: Detects craters in images by segmenting them from the background.
- **Pre-trained Model**: The model is initialized with pre-trained weights, making it suitable for transfer learning.
- **Customizable Workflow**: Users can customize training, testing, and validation processes.
- **Data Handling**: The dataset includes labeled images in YOLO format for training and testing the model.

## Dataset
The dataset used in this project consists of lunar crater images and their corresponding label files in YOLO format. The labels contain the class IDs and bounding box coordinates for each crater detected in the images.

You can download the dataset from [Kaggle - Martian Lunar Crater Detection Dataset](https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset).

### File Structure
The dataset is organized as follows:
```
/kaggle/input/martianlunar-crater-detection-dataset/craters/
    ├── train/
    │   ├── images/  # Training images
    │   └── labels/  # Corresponding labels in YOLO format
    ├── valid/
    │   ├── images/  # Validation images
    │   └── labels/  # Corresponding labels in YOLO format
    └── test/
        ├── images/  # Test images
        └── labels/  # Corresponding labels in YOLO format
```

## Installation

### Requirements
To run this project, you need to have the following installed:

- Python 3.7+
- TensorFlow (2.x)
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Keras (if separate from TensorFlow)
  
You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### File Setup
Ensure that you have the following files:
- `train/images/` (contains training images)
- `train/labels/` (contains corresponding labels in YOLO format)
- `valid/images/` (contains validation images)
- `valid/labels/` (contains corresponding labels in YOLO format)
- `test/images/` (contains test images)
- `test/labels/` (contains corresponding labels in YOLO format)

## Usage

### 1. Loading the Data
The data is loaded using the `load_data` function, which loads both images and corresponding labels from the specified directories.

```python
train_images, train_masks = load_data(train_img_path, train_lbl_path)
valid_images, valid_masks = load_data(valid_img_path, valid_lbl_path)
test_images, test_masks = load_data(test_img_path, test_lbl_path)
```

### 2. Training the Model

The model is trained using the U-Net architecture. The training process involves fitting the model to the training data and validating it on the validation set.

```python
model = build_unet_model(input_size=(128, 128, 3))
model.fit(train_images, train_masks, validation_data=(valid_images, valid_masks), epochs=10, batch_size=8)
```

**Additional Options for Training**:
- **Custom Batch Size**: Adjust the `batch_size` parameter based on available hardware.
- **Epochs**: Control the number of training epochs.
- **Learning Rate**: You can customize the learning rate in the optimizer.

```python
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

### 3. Testing the Model

After training, the model is used to make predictions on the test dataset. The predicted masks are compared to the true labels to evaluate the model.

```python
predicted_masks = model.predict(test_images)
```

### 4. Saving the Model

Once the model is trained, it can be saved for future use.

```python
model.save(model_save_path)
```

### 5. Customizing Image Augmentation (Optional)

You can apply image augmentations (e.g., rotations, flipping, scaling) to the training data to improve the model's robustness. Modify the data pipeline as follows:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen = datagen.flow(train_images, train_masks, batch_size=8)
```

### 6. Visualizing Results

You can visualize the model's predictions using `matplotlib`:

```python
import matplotlib.pyplot as plt

# Display first image and its prediction
plt.subplot(1, 2, 1)
plt.imshow(test_images[0])
plt.title("Test Image")

plt.subplot(1, 2, 2)
plt.imshow(predicted_masks[0], cmap='gray')
plt.title("Predicted Mask")
plt.show()
```

## Model Architecture

This project uses the **U-Net** architecture, which is a popular deep learning model for image segmentation. It consists of a contracting path (encoder) and an expanding path (decoder) with skip connections to retain high-resolution features.

### U-Net Architecture Overview:
- **Encoder**: Downsampling layers to extract feature maps.
- **Bottleneck**: The deepest layer where feature extraction is most concentrated.
- **Decoder**: Upsampling layers that restore spatial resolution while leveraging encoder features via skip connections.
  
The final output is a binary mask where each pixel represents the presence of a lunar crater.

## Additional Options

### 1. Model Evaluation (without metrics)

You can evaluate the model without metrics (e.g., loss) using:

```python
model.evaluate(test_images, test_masks)
```

### 2. Fine-tuning Pre-trained Model

You can fine-tune a pre-trained model by unfreezing some of the layers and training it on your dataset.

```python
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_masks, validation_data=(valid_images, valid_masks), epochs=5)
```

### 3. Model Inference on New Images

If you have new images to test (not in the dataset), you can run inference on them as follows:

```python
new_images = load_new_images('/path/to/new/images')
predictions = model.predict(new_images)
```

### 4. Hyperparameter Tuning

To improve model performance, you can experiment with different hyperparameters such as:
- Learning rate
- Number of layers/units in the U-Net architecture
- Dropout rate
- Batch size

You can use `GridSearchCV` or `RandomizedSearchCV` from `sklearn` to perform hyperparameter tuning.