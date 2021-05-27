# Fish box
This repo features an image processing algorithm that undistorts images along with segmentation of cod and classification of damage to cod.

The input is pictures of cod and the output is an image where wound on cod is marked and tagged along with a CDI (catch damage index).
The CDI marks which image file names that contains cod with wounds.

## How to use
1. Put cod images into fish_pics/input_images.

2. Select the run mode by setting the script parameters in the configurations or passing them in command line/terminal/shell. [How to pass arguments for a python script](https://www.youtube.com/watch?v=m8MkQmrJdzk) :    
    - "n" : Run the program using only openCV functions
    - "y" : Run the program using own built functions
    - "y e": Run the program using own built functions, and see not yet finished version of future implementations

    NOTE: when running the program using own built functions it normally takes 420 sec. for one image. Consider having only few images in fish_pics/input_images

3. The output is saved.
    1. Images for manual inspection is saved in fish_pics/output_images/manual_inspection
    2. Images for manual inspection with CLAHE applied is saved in fish_pics/output_images/manual_inspection_CLAHE
    3. Images where wounds are marked are saved in fish_pics/output_images/marked_images
    4. The CDI is saved in output_CDI

## Reference ode overview

1. [Grass fire algorithm](link)
2. [Harris corner detection](link)
3. [Thresholding](link)
4. [Find center of contur mass](link)
5. [Camera calibration](link)
6. [Blur](link)
***
### Morphology
7. [Erosion](link) 
8. [Dialation](link)
9. [Opening](link)
10. [Closing](link)
11. [Sobel operator](link)
***
### Color converting
12. [Gray scaling](link)
13. [RGB to HSV](link)
