# Dog-Breed-Classification

This project was developed as a part of Udacity's Deep Learning Nanodegree. In this project, i have created a convolutional neural network from scratch using pytorch and also created a cnn using transfer learning in pytorch.

# Getting Started

Just run the ipynb notebook. Tune the hyper parameters for better accuracy.

## Ouput

<img src="/images/human.jpg" width=300px>     <img src="/images/dog.jpg" width=300px> 

## Download

Download dog image file from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

Download dog image file from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

### Prerequisites

* Python 3.
* Numpy 
* Pandas
* MatPlotLib
* OpenCv
* Pytorch. 

### Install

1. Clone the repository and navigate to the downloaded folder.
	```	
	git clone https://github.com/vickipedia6/Dog-Breed-Classification.git
	cd Dog-Breed-Classifier
	```
2. Open the jupyter notebook
	```
	jupyter notebook Dog-breed_classifier.ipynb	
	```
3. Load the pre trained model using the following code in jupyter notebook
    <code>
       model_scratch.load_state_dict(torch.load('model_scratch.pt'))
    </code>

### Architecture

 <img src='nn (1).svg' width=500px> 

### Project results

This project met the following specifications:
* The submission includes all required, complete notebook files.
* The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.
* Used a pre-trained VGG16 Net to find the predicted class for a given image. Used this to complete a dog_detector function below that returns True if a dog is detected in an image (and False if not).
* The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected dog.
* Wrote three separate data loaders for the training, validation, and test datasets of dog images. These images were pre-processed to be of the correct size.
* Answer describes how the images were pre-processed and/or augmented.
* The submission specifies a CNN architecture.
* Answer describes the reasoning behind the selection of layer types.
* Choosing appropriate loss and optimization functions for this classification task. Trained the model for a number of epochs and saved the "best" result.
* The trained model attains at least 10% accuracy on the test set.
* The submission specifies a model architecture that uses part of a pre-trained model.
* The submission details why the chosen architecture is suitable for this classification task.
* Train your model for a number of epochs and save the result wth the lowest validation loss.
* Accuracy on the test set is 60% or greater.
* The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN.
* The submission uses the CNN from the previous step to detect dog breed. The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.
* The submission uses the CNN from the previous step to detect dog breed. The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.
* Submission provides at least three possible points of improvement for the classification algorithm.

## Losses

### Model scratch:
Training loss: 4.158 ... Validation loss: 3.442

### Transfer model:
Training loss: 0.0951 ... Validation loss: 0.025 

## Probable Errors

During Training this error can occur "OSError: image file is truncated (49 bytes not processed)". Use this code to solve it

<code>
 from PIL import ImageFile /n
 ImageFile.LOAD_TRUNCATED_IMAGES = True
</code>

## Accuracy:

### Model scratch:
Accuracy : 14%

### Transfer model:
Accuracy : 79%

## Built With

* Python 3

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* The data comes from Udacity.
