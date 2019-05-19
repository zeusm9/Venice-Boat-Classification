# Venice-Boat-Classification
Image classifier for Venice boat recognition

Detection and classification of vehicles are fundamental processes in moni-
toring and security systems. In this report we focus on a particular catgory
of vehicles : boats. Monitoring a maritme scene implies a series of challenges
related for example to the water background, re
ections or to the light in the
images. All this features of the image make the problems of boat detection
and classification an hard work.
Our work is focussed on boat images automatically extracted by the ARGOS
system (Automatic and Remote GrandCanal Observation System) operating
24/7 in Venice. The system automatically extracts information on the traffic
flow and density and it highlights the illegal behaviour of the boats driver.
ARGOS controls a water canal of about 6 km length, with a width between
80 and 150 meters. Cameras are connected to a dedicated computer where
high resolution color images are acquired and processed.
ARGOS provides the following main functionalities:
- Optical detection and tracking of moving targets present in the field of
view of each survey cell;
- Computing position, speed, and heading of any moving target observed
by a cell;
- Automatic detection of a set of predened events;
- Transmission of data and video stream to the Control Center.

Given the dataset provided by MarDCT (Maritime detection, classification
and tracking dataset) of the Venice boats images, build an image classifier
to classify the boats according to the type of boat. To solve the problem it
has been used a convolutional neural network approach with preprocessing
of the images.
After building multiple models with different parameters in the convolutional
neural network, the model has been tested on a validation dataset still provided by MarDCT.
