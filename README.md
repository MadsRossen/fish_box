# Harris corner detection

## Math behind it:

Wiki: [Corner detection](https://en.wikipedia.org/wiki/Corner_detection)

Wiki: [Harris corner detection](https://en.wikipedia.org/wiki/Harris_Corner_Detector)

PDF1: [Cornell unversity](http://www.cs.cornell.edu/courses/cs4670/2015sp/lectures/lec07_harris_web.pdf)

PDF2: [DIGI VFX](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/lectures/handouts/lec06_feature_4up.pdf)

PDF3:[Tayloe series/jacobian math](http://www.cs.cmu.edu/~16385/s17/Slides/14.4_Alignment__LucasKanade.pdf)

PDF: [DIGI VFX](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/lectures/handouts/lec06_feature_4up.pdf)

Youtube: [Interest point detection](https://www.youtube.com/watch?v=_qgKQGsuKeQ)

## Code inspiration:

Github: [adityaintwala](https://github.com/adityaintwala/Harris-Corner-Detection)

## Parameters to change
windowsize is only used for the shifting window offset, and is not equal to the actual size of the shifting window. [here](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L11)
```python
windowSize = 3 
``` 
Harris corner constant [here](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L12)
```python
k = 0.04 # Parameter between 0.04 - 0.06
```
Threshold is normally around 10000 [here](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L13)
```
threshold = 10000
```
Size of the blur [here](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L28)
```python
blur = cv.blur(filename, (3, 3))
```
Number of points at the checkerboard. Ex. 9 x 6 = 54 points. Can be changed [here](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L15)
```python
CheckPoints = 54
```
## Create the Harris Response and image gradient distribution
Run the [Matlab script](https://github.com/MadsRossen/fish_box/blob/detection/HarrisResponse.m)
and use the text file createt form [harris-corner-detection.py](https://github.com/MadsRossen/fish_box/blob/detection/harris-corner-detection.py). Remember to set the boolean [test](https://github.com/MadsRossen/fish_box/blob/f1947c68d0196c51ed7bfd74ac125ea4fe3149b0/harris-corner-detection.py#L14) to True.

Else:
```python
test = False
```
# Sobel–Feldman operator

## Math 

Wiki: [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)

More: [Sobel Edge Detector](https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm)
