# Dog Breed Classification Project

## Overview

This project focuses on the classification of 70 dog breeds using Convolutional Neural Networks (CNNs). We utilized three different models—EfficientNetB0, ResNet50, and YOLOv8—to fine-tune baseline architectures on a Kaggle dataset of 70 dog breeds. The goal was to develop a robust system capable of accurately identifying the breed of a dog from an image.

## Dataset

The dataset used for this project is the "70 Dog Breeds-Image Data Set" sourced from Kaggle. It contains over 9,400 high-resolution images of 70 different dog breeds. The dataset is divided into training, testing, and validation sets as follows:

- **Training Set:** 7,946 images
- **Testing Set:** 700 images
- **Validation Set:** 700 images

Each image is labeled with its corresponding dog breed, enabling the models to learn the distinct patterns and characteristics of each breed.

## Models Used

### 1. EfficientNetB0
EfficientNetB0 is known for its efficiency in terms of parameter size and computational requirements. We used a pre-trained EfficientNetB0 model from Keras, fine-tuning it with additional dense layers to classify the 70 breeds.

### 2. ResNet50
ResNet50 is a deeper network that includes residual connections to effectively handle the vanishing gradient problem. We fine-tuned a pre-trained ResNet50 model from Keras, similarly adding dense layers for the classification task.

### 3. YOLOv8
YOLOv8 is primarily an object detection model, but we adapted it for the classification task by leveraging its strong feature extraction capabilities. YOLOv8 proved to be the most efficient model in our experiments, providing a good balance between performance and computational efficiency.

## Results

The results of our models on the test set are summarized below:

| Model         | Precision | Recall | F1-Score | Accuracy | Params | Params Memory |
|---------------|-----------|--------|----------|----------|--------|---------------|
| EfficientNetB0| 0.92      | 0.91   | 0.90     | 0.91     | 5.3M   | 20.22MB       |
| ResNet50      | 0.89      | 0.88   | 0.87     | 0.88     | 26.7M  | 101.8MB       |
| YOLOv8        | 0.96      | 0.95   | 0.95     | 0.96     | 1.5M   | 5.72MB        |

YOLOv8 outperformed the other models in all metrics, making it the top choice for this classification task.

## Usage

Firstly, you need to download the data from this [link](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)
. After it is downloaded, place it in models directory structured as is (dataset/...). 
To train models, just run each notebook. Additional instructions are provided as comments in notebooks.

## Paper

For a detailed explanation of our methodology, experimental setup, and analysis, please refer to our research paper attached in this repository.

## Conclusion

This project highlights the importance of model selection and fine-tuning in achieving high accuracy in image classification tasks. YOLOv8's superior performance demonstrates its effectiveness in balancing model complexity with computational efficiency, making it a strong candidate for real-world applications.
