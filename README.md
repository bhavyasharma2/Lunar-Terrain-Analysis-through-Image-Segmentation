# Lunar Terrain Analysis through Image Segmentation

This project was developed as part of my Machine Learning Research Traineeship at Spartificial. It focuses on lunar terrain analysis using deep learning-based image segmentation to detect and classify surface features like rocks. The project implements U-Net, a widely used convolutional neural network (CNN) architecture for semantic segmentation, to process high-resolution lunar imagery.

## Project Overview

Understanding the composition and structure of the lunar surface is crucial for space exploration, rover navigation, and mission planning. This project utilizes computer vision and deep learning techniques to enhance rock detection and segmentation in lunar images.

The primary objective is to train a robust image segmentation model that can accurately distinguish between rock formations and background terrain, enabling better scientific analysis and automated exploration.

## Key Features

- **Semantic Segmentation with U-Net:** A fully convolutional neural network (FCN) optimized for pixel-wise classification.
- **Deep Learning Framework:** Uses TensorFlow and Keras for model training and optimization.
- **Efficient Image Processing:** Preprocessing techniques using NumPy and Matplotlib.
- **High Accuracy Metrics:** Achieved a maximum IoU (Intersection over Union) score of 0.80, indicating precise segmentation.


## Setup Instructions

To set up your development environment and run this project locally, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bhavyasharma2/Lunar-Terrain-Analysis-through-Image-Segmentation
   cd Lunar-Terrain-Analysis-through-Image-Segmentation
   ```

2. **Create and Activate a Virtual Environment:**
   - **For Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **For macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the Required Dependencies:**
   - Upgrade pip:
     ```bash
     pip install --upgrade pip
     ```
   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Project:**
   - Make sure the dataset is extracted in the correct directory (`data/lunar_dataset`).
   - Then run:
     ```bash
     python main.py
     ```

> If you encounter any issues, double-check that the dataset paths in `main.py` are accurate and that you're running inside the virtual environment.


## Dataset

We used the **Artificial Lunar Rocky Landscape Dataset** from Kaggle. To replicate this project, please follow the steps below to download and set up the dataset:

1. **Download the Dataset:**
   - Go to the Kaggle dataset page: [Artificial Lunar Rocky Landscape Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset).
   - Click on the "Download" button to get the dataset as a `.zip` file.

2. **Place the Dataset in the Project:**
   - After downloading, rename the downloaded folder to `lunar_dataset.zip` and move this `lunar_dataset.zip` folder to `data/` directory.

3. **Extract the Dataset:**
   - The code to extract contents of `lunar_dataset.zip` is already provided in `main.py`, The folder should contain subdirectories like `images/render` and `images/clean`.

   Your directory structure should look like this:
   ```
   project/
   ├── data/
   │   └── artificial-lunar-rocky-landscape-dataset/
   │       ├── images/
   │       │   ├── render/
   │       │   └── clean/
   ├── main_script.py
   ├── README.md
   ```

4. **Update Dataset Path in Code:**
   - In the `main.py` file, ensure the paths to the images and masks are set as follows:
     ```python
     img_dir = 'data/lunar_dataset/images/render'
     mask_dir = 'data/lunar_dataset/images/clean'
     ```

After completing these steps, you should be ready to run the project.



## Methodology

### Data Preprocessing:

- Loaded grayscale lunar images and corresponding segmentation masks.
- Applied normalization and augmentation techniques to improve generalization.

### Model Architecture (U-Net Implementation):

- Used U-Net for precise pixel-wise classification.
- Implemented skip connections to retain spatial information.
- Optimized using the Adam optimizer with binary cross-entropy loss.

### Training & Evaluation:

- Trained the model with a dataset of labeled lunar images.
- Evaluated using IoU (Intersection over Union) and Dice Coefficient.
- Achieved 0.80 IoU score, demonstrating high segmentation accuracy.

## 🛠Technologies Used

This project leverages the following technologies:

| Technology      | Purpose                                        |
|-----------------|------------------------------------------------|
| Python          | Primary programming language for model development. |
| NumPy           | Numerical computation and efficient array handling. |
| Matplotlib      | Data visualization, used for plotting images and model performance. |
| TensorFlow      | Deep learning framework for defining and training neural networks. |
| Keras           | High-level API for TensorFlow, simplifies model building. |
| Jupyter Notebook| Interactive development environment for running experiments. |

> 📓 This project was initially developed as a Jupyter Notebook, which is also included in this repository for reference.  
> You can find it as [`Image_Segmentation_of_Lunar_Surface_Images.ipynb`](Image_Segmentation_of_Lunar_Surface_Images.ipynb).

## Future Enhancements

- **Hyperparameter Optimization:** Fine-tune learning rates, batch sizes, and dropout rates for better performance.
- **Transfer Learning:** Experiment with ResNet-based encoders for feature extraction.
- **Advanced Augmentation:** Use Albumentations or OpenCV for enhanced preprocessing.
```
