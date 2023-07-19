# Pedestrians Detection and Tracking
This project aims to detect and track pedestrians using a pretrained MobileNet SSD (Single Shot MultiBox Detection) model. The MobileNet SSD model is a lightweight deep learning model that combines the MobileNet architecture and SSD framework, making it suitable for efficient object detection tasks.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- argparse

## Getting Started

To get started with the project, follow the steps below:

1. Clone the repository to your local machine:

```shell
git clone https://github.com/VDai1999/ObjDetectionAndTracking.git
```

2. Change to the project directory:

```shell
cd ObjDetectionAndTracking
```

3. Download Dataset (e.g, Oxford Town Centre Dataset, etc.)
Link: https://academictorrents.com/details/35e83806d9362a57be736f370c821960eb2f2a01

4. Download MobileNet SSD Model
Link: https://drive.google.com/drive/folders/14IM5aiiIZ8DTPgZMrSBuL7lXoPca-08g?usp=drive_link

5. Run the `objDetectTrack.py` script:

```shell
python objDetectTrack.py TownCentreXVID.avi
```

This script will utilize the MobileNet SSD model to detect pedestrians in a video file. Detected pedestrians will be highlighted in bounding boxes.

## Acknowledgements

- The MobileNet SSD model is based on the work by Howard, A. et al. (2017), "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
- The project was inspired by various pedestrian detection and tracking implementations available in the computer vision community.
