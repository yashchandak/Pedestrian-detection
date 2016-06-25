# Pedestrian-detection

Pedestrian detection on **Raspberry Pi** using **Neural Nets**

##Features

- **Trainable for other custom objects instead of Pedestrians**
- Real-time detection 

  ###Dataset features
  - TLD tracker(from OpenCV) can be used to extract cutom objects from videos, (requires C++)
  - Images can also be augmented (flipped/brightness/contrast) to increase dataset size


##Working
- Train the system on the PC
- Copy the saved weights onto the Pi
- Two threads execute during runtime
    - One reads images from camera 
    - Second performs detection 
- To speed up computation, moving objects are exctracted using difference of frames
- Extracted objects are passed to neural net for classification
- If it is the required object, it is marked and displayed


##Dependencies

- Python
- Numpy
- OpenCV

##Sample Result
[Image](/Report/result.png)
