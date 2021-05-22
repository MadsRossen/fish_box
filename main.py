import time

import cv2

# Imports of functions where own built functions have been used:
from extremeImageProcessing import saveCDI
# Imports of functions where openCV functions have been used:
from functions_openCV import checkerboard_calibrateOPENCV, detect_bloodspotsOPENCV, segment_codOPENCV, showSteps, \
    save_imgOPENCV, crop, loadImages
from yamlLoader import yaml_loader, setup_parameters

'''User options'''

# Run algorithm with opencvV or own built functions
openCV = True

# Choose which fish the steps are shown for
showFish = 0

'''User options'''

# Run program using openCV functions
if openCV:
    stepsList = []

    # Start timer for algorithm
    start_time = time.time()

    # Load in yaml data from the file
    yaml_data = yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    kernels, checkerboard_dimensions, paths, clahe = setup_parameters(yaml_data, printParameters=False)

    # Load Fish images
    images, names, img_list_fish = loadImages(paths[0][1], False, False, 40)
    stepsList.append(images[showFish])

    # Load checkerboard images
    img_cali, names_cali, _ = loadImages(paths[1][1], False, False, 40)

    # Calibrate camera and undistort images
    fish_cali = checkerboard_calibrateOPENCV(checkerboard_dimensions, images, img_cali,
                                             show_img=False, recalibrate=False)
    stepsList.append(fish_cali[showFish])

    # Crop to ROI
    cropped_images = crop(fish_cali, 710, 200, 720, 2500)
    cv2.imwrite('fish_pics/segment_this.JPG', cropped_images[5])
    stepsList.append(cropped_images[showFish])

    # Threshold to create a mask for each image
    mask_cod, img_segmented_cod = segment_codOPENCV(cropped_images)
    stepsList.append(img_segmented_cod[showFish])

    # Blood spot detection
    mask_bloodspots, bloodspots, marked_bloodspots, \
    percSpotCoverage = detect_bloodspotsOPENCV(img_segmented_cod, mask_cod)
    stepsList.append(bloodspots[showFish])
    stepsList.append(marked_bloodspots[showFish])

    # Save marked blood spots images in folder
    save_imgOPENCV(marked_bloodspots, 'fish_pics/output_images', img_list_fish)

    # Save a .txt file with CDI (catch damage index)
    saveCDI(img_list_fish, percSpotCoverage)

    # Check how long it took for algorithm to finish
    end_time = time.time()
    print('Total time for algoritm: ', end_time - start_time, 'sec')

    # Create subplots of main steps
    showSteps(stepsList)

# Run program using own built functions
if openCV == False:
    0
