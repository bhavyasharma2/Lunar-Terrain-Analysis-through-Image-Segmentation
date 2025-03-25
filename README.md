# Lunar Terrain Analysis through Image Segmentation

This project was developed as part of my **Machine Learning Research Traineeship at Spartificial**. It focuses on **lunar terrain analysis** using **deep learning-based image segmentation** to detect and classify surface features like rocks. The project implements **U-Net**, a widely used convolutional neural network (CNN) architecture for semantic segmentation, to process high-resolution lunar imagery.

---

## Project Overview

Understanding the **composition and structure** of the lunar surface is crucial for space exploration, rover navigation, and mission planning. This project utilizes **computer vision** and **deep learning** techniques to enhance rock detection and segmentation in lunar images.

The primary objective is to **train a robust image segmentation model** that can accurately distinguish between **rock formations and background terrain**, enabling better **scientific analysis and automated exploration**.

## Key Features

- **Semantic Segmentation with U-Net**: A **fully convolutional neural network (FCN)** optimized for pixel-wise classification.
- **Deep Learning Framework**: Uses **TensorFlow** and **Keras** for model training and optimization.
- **Efficient Image Processing**: Preprocessing techniques using **NumPy** and **Matplotlib**.
- **High Accuracy Metrics**: Achieved a **maximum IoU (Intersection over Union) score of 0.80**, indicating precise segmentation.

---

## Dataset

We used the **Artificial Lunar Rocky Landscape Dataset** from Kaggle. To replicate this project:

1. **Download** the dataset from Kaggle.
2. **Set the dataset path** in the code (`img_dir` and `mask_dir` for images and segmentation masks).
3. **Run the Jupyter Notebook** to train and evaluate the model.

---

## Methodology

1. **Data Preprocessing**:  
   - Loaded grayscale lunar images and corresponding segmentation masks.
   - Applied **normalization** and **augmentation** techniques to improve generalization.

2. **Model Architecture (U-Net Implementation)**:
   - Used **U-Net** for precise **pixel-wise classification**.
   - Implemented **skip connections** to retain spatial information.
   - Optimized using the **Adam optimizer** with **binary cross-entropy loss**.

3. **Training & Evaluation**:
   - Trained the model with a dataset of labeled lunar images.
   - Evaluated using **IoU (Intersection over Union) and Dice Coefficient**.
   - Achieved **0.80 IoU score**, demonstrating high segmentation accuracy.

---

## 🛠Technologies Used

This project leverages the following technologies:

| Technology    | Purpose |
|--------------|---------|
| **Python**   | Primary programming language for model development. |
| **NumPy**    | Numerical computation and efficient array handling. |
| **Matplotlib** | Data visualization, used for plotting images and model performance. |
| **TensorFlow** | Deep learning framework for defining and training neural networks. |
| **Keras**    | High-level API for TensorFlow, simplifies model building. |
| **Jupyter Notebook** | Interactive development environment for running experiments. |

---

## Future Enhancements

- **Hyperparameter Optimization**: Fine-tune **learning rates, batch sizes, and dropout rates** for better performance.
- **Transfer Learning**: Experiment with **ResNet-based** encoders for feature extraction.
- **Advanced Augmentation**: Use **Albumentations** or **OpenCV** for enhanced preprocessing.
- **Real-World Application**: Train the model on **actual lunar images** from NASA datasets for improved generalization.
