# Deep Learning

## Preprocessing
After loading all the images along with their corresponding targets, the data is randomly split into train, validation, and test sets. These sets are then loaded into data generators, which will be used to feed the neural network. The target images are thresholded, with any pixel value greater than or equal to 128 denoted as a positive label indicating the occurrence of blood vessels.

## U-Net
U-Net is a popular deep neural network architecture known for effectiveness in semantic segmentation tasks. By utilizing convolutional layers and residual connections, U-Net achieves exceptional performance in capturing complex details. It has been widely used as a preferred solution for state-of-the-art semantic segmentation challenges, which is why it was chosen for this task. For more information, I refer you to [scientific paper](https://arxiv.org/pdf/1505.04597v1.pdf]).

## Training
I have experimented with various optimizers, including RMSprop, SGD, and different variants of Adam, using different learning rates. When stability was observed, I experimented with two approaches:
- training from the beginning on the combined train and validation set
- retraining already trained model on the validation set

 Afterward, the results were comparable for each approach.

 Accuracy and loss plots:

![image](https://github.com/kamil271e/retinal-vessels-segmentation/assets/82380348/081ef5cf-a1bc-4b4f-aec5-9c856ee7330b) | ![image](https://github.com/kamil271e/retinal-vessels-segmentation/assets/82380348/302f1531-f9d4-4ced-91ae-b12fc5130f97)
:---------------------:|:---------------------:

## Results
After above mentioned steps, model performance was sustainable during different random splits of data and established approximetaly to 0.8 on precision, recall and F1-Score metrics with accuracy up to 0.98.

![image](https://github.com/kamil271e/retinal-vessels-segmentation/assets/82380348/aba89212-fd2d-477e-92fd-1bce2f78ffe9)


Example classification report which shows the result for test set that consist of 5 images of resolution 512x512:

|    Class    | Precision |  Recall  | F1-Score | Support |
|:-----------:|:---------:|:--------:|:--------:|:-------:|
|      0      |    0.98   |   0.98   |   0.98   |  1211630 |
|      1      |    0.78   |   0.79   |   0.78   |   99090  |
|             |           |          |          |         |
|  Accuracy   |           |          |   0.97   | 1310720 |
|  Macro Avg  |    0.88   |   0.88   |   0.88   | 1310720 |
|Weighted Avg |    0.97   |   0.97   |   0.97   | 1310720 |



