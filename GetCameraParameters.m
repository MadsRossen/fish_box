clear all

% Load image for showing distortion effect.
imageFileNames = {'.\checkerboard_pics\GOPR1621.JPG'};

% Load image points.
imagePointsOrigload = load('imgPoints.mat', '-mat');
imagePointsOrig = double(imagePointsOrigload.np_array_switch(:,:,1:4));
imagePointsOrig(:,:,end+1) = double(imagePointsOrigload.np_array_switch(:,:,6));

% Define the total amount of white and black squares on the checkerboard [Height, Width]. 
boardSize = [7,10];

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 25;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);


% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePointsOrig, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% Show undistorted image for reference.
imshow(undistortedImage)

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Print intrinsic parameters and radial distortion coeffiecients.
IntrinsicMatrix = cameraParams.IntrinsicMatrix

RadialDistortion = cameraParams.RadialDistortion
