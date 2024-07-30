# CNN Implementation for CIFAR-10 Image Classification




This repository contains the implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset with TensorFlow and Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 50,000 images for training and 10,000 for testing.

### Project Overview

1. **Importing Libraries**:
   - TensorFlow and Keras for building and training the neural network.
   - Matplotlib for plotting images and training results.

2. **Loading and Preprocessing Data**:
   - Load the CIFAR-10 dataset.
   - Normalize pixel values to be between 0 and 1.

3. **Visualizing the Data**:
   - Plot the first 25 images from the training set along with their class names to verify the dataset.

4. **Creating the Convolutional Base**:
   - Define a sequential model with a stack of Conv2D and MaxPooling2D layers.
   - Configure the input shape to process images of shape (32, 32, 3).

5. **Adding Dense Layers**:
   - Flatten the output from the convolutional base to a 1D vector.
   - Add Dense (fully connected) layers to perform the classification. The final Dense layer has 10 outputs, corresponding to the 10 classes in CIFAR-10.

6. **Compiling and Training the Model**:
   - Compile the model using the Adam optimizer and SparseCategoricalCrossentropy loss function.
   - Train the model for 10 epochs with the training data and validate with the testing data.

7. **Plotting Training Results**:
   - Plot training and validation accuracy to visualize the training process and assess the model's performance over epochs.

8. **Evaluating the Model**:
   - Evaluate the model on the test data to determine its accuracy , This simple CNN has achieved a test accuracy of over 70% ..
