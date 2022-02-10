# Bird Watcher
The project was the result of experimenting with machine learning with the end goal of deploying something onto a Raspberry Pi. The reason for birds is just that it was the first and easiest to use free dataset that I could also realistically test with a camera. 

## Technologies Used and Explored

- TensorFlow (and TFlite)
- YOLOv5 (Pytorch) 
- Docker

## Components
### Data Prep and Training
The Jupyter notebook (makeMode.ipynb) contains all necessary commands to retrieve and prepare the Caltech 2011 Birds dataset for YOLOv5. 

I originally created it in Google Colab and that is where many of the image producing and plotting functions work correctly, but the functionality works across a variety of Jupyter notebook platforms. 

#### Docker
Due to the size of the dataset and time required to train the model, I first looked into training the model on my local computer using Docker. The docker system works and can be easily modified to run with a local Nvidia GPU, but I don't have one, and CPU training times are unusably long. 

One note about Docker: local systems often have issues with Yolov5's workers. Disable them with `--worker 0`. CPU training can be similarly configured with `--device cpu`.

#### Saturn Cloud
I chose Saturn cloud as an alternative to Colab for training my model. The 30 hours of free GPU time per month are more than enough to train with a couple hundred epochs. 

### Deployment
The deployment folder contains everything necessary for deploying this model on a Raspberry Pi with a USB webcam as I did. It contains some trained models as well. I run mine in a python virtual environment with 
```
python3 bw-deploy.py -m best-fp16.tflite
```

## Credits/Links
[YOLOv5](https://github.com/ultralytics/yolov5)

[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

[TensorFLow Detection Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py#L117)

[EdjeElectronic's Detection Example](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_webcam.py)