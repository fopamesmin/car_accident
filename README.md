# Car Accident Detection from Images

## Project Overview
This project aims to develop a deep learning model to classify images of cars into two categories: 'accidented' and 'nonaccident'. The model is trained using Convolutional Neural Networks (CNNs) and is capable of distinguishing between images of cars involved in accidents and those that are not.

## Dataset Description
The dataset for this project consists of two folders:
- **accidented**: Contains images of cars involved in accidents.
- **nonaccident**: Contains images of cars that are not involved in accidents.

All images are in `.jpg` format. The images are preprocessed to a standard size and normalized before being used for training the model.

## Model Architecture
The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:
- **Conv2D Layer**: 32 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Conv2D Layer**: 64 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Conv2D Layer**: 128 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Flatten Layer**
- **Dense Layer**: 128 units, activation 'relu'
- **Dropout Layer**: dropout rate 0.5
- **Dense Layer**: 2 units, activation 'softmax'

The model is compiled with the Adam optimizer and uses categorical cross-entropy as the loss function. Accuracy is used as the evaluation metric.

## Training Process
The model is trained using images from both classes ('accidented' and 'nonaccident'). Data augmentation techniques such as rotation, width shift, height shift, and horizontal flip are used to enhance the training data and improve the model's generalization capability.

The training process involves:
1. Loading and preprocessing the images.
2. Splitting the dataset into training and testing sets.
3. Building the CNN model.
4. Training the model using the training data.
5. Evaluating the model using the testing data.
6. Saving the trained model for future use.

## Results
- Training accuracy: 95%
- Testing accuracy: 90%
- The model demonstrates good performance in distinguishing between accidented and nonaccident images.

## Usage
To use the trained model for predicting whether a car image is accidented or nonaccident, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Amrampro/car-accident-detection.git
    cd car-accident-detection
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Load the Model**:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('car_accident_detection_model.h5')
    ```

4. **Preprocess the Image**:
    ```python
    import cv2
    import numpy as np
    
    def preprocess_image(image_path, target_size=(128, 128)):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    ```

5. **Make a Prediction**:
    ```python
    test_image_path = 'path_to_test_image.jpg'  # Replace with your test image path
    test_image = preprocess_image(test_image_path)
    prediction = model.predict(test_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    classes = ['nonaccident', 'accidented']
    predicted_label = classes[predicted_class]
    print(f'The model predicts this image is: {predicted_label}')
    ```

6. **Visualize the Result**:
    ```python
    import matplotlib.pyplot as plt
    
    plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {predicted_label}')
    plt.axis('off')
    plt.show()
    ```

## Future Work
- Increase the dataset size for better generalization.
- Experiment with different model architectures.
- Use transfer learning techniques.
- Implement real-time detection in a web or mobile application.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
