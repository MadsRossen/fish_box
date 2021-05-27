import sys
import time

import extremeImageProcessing as eip
import functions as ft
import functions_openCV as ftc
import yamlLoader as yamlL
from functions_openCV import claheHSL

# Check if user have passed arguments
try:
    sys.argv[1]
except IndexError as ie:
    print(
        "You need to pass arguments when running the script \n \'n\' - for openCV \n \'y\' - for Group 460 built \n \'y"
        "e\' - for Group 460 with extra non finished functions")

run_own_functions = sys.argv[1]
experimental = 0

# Does the user want to use our own functions or OpenCV functions. y for own functions, n for openCV
# Note for one cod image it takes around 420 sec
if run_own_functions == 'y':

    print("When running own built it normally takes 420 sec for 1 image (8 min)")
    print("Therefore you might want to only have one image in fish_pics/input_images")
    print("Running program with own built-in functions")

    # Check the runtime
    start_time = time.time()

    # Create list with images showing the different main steps of the algorithm
    stepsList = []
    showFish = 0

    # First check if the software should run the experimental
    if len(sys.argv) > 2:
        experimental = sys.argv[2]
        # debug = sys.argv[3]

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    _, paths, clahe, cali_pa = yamlL.setup_parameters(yaml_data)

    # load Fish images
    images, img_list_fish = ft.loadImages(paths[0][1], False, False, 60)
    stepsList.append(images[showFish])

    # Calibrate camera and undistort images
    fish_cali = eip.undistort(images, cali_pa[0][1], cali_pa[1][1], cali_pa[2][1], cali_pa[3][1], cali_pa[4][1],
                              cali_pa[5][1], show_img=False)
    stepsList.append(fish_cali[showFish])

    # Crop to ROI to make the processing time smaller
    cropped_images = eip.crop(fish_cali, 710, 200, 720, 2500)
    stepsList.append(cropped_images[showFish])

    # Threshold to create a mask for each image
    masks, segmented_images = eip.segment_cod(cropped_images, False)

    # Use morphology on images
    images_morph, res_images = ft.morphology_operations(masks, segmented_images, 5, 7, False, False)
    stepsList.append(res_images[showFish])

    # Wounds spot detection
    masks_woundspot, woundspots, marked_woundspots_imgs, _, damage_percentage = ft.detect_woundspots(res_images)
    stepsList.append(woundspots[showFish])
    stepsList.append(marked_woundspots_imgs[showFish])

    # Save marked wounds images in folder
    # CLAHE images
    ftc.save_imgOPENCV(segmented_images, 'fish_pics/output_images/manual_inspection', img_list_fish,
                       "_MANUAL_INSPECTION")

    # Marked images
    ftc.save_imgOPENCV(marked_woundspots_imgs, 'fish_pics/output_images/marked_images', img_list_fish, "_marked")

    # Save a .txt file with CDI (catch damage index)
    ftc.saveCDI(img_list_fish, damage_percentage)

    # If the user wanna try the experimental features of the program
    if experimental == 'e':
        # Get the contours
        contour = ft.find_contours(masks, cropped_images, False, False)

        # Find contours middle point
        xcm, ycm = ft.contour_MOC(cropped_images, contour)

        # Raytracing
        rot_img = ft.raytracing(cropped_images, xcm, ycm, contour)

    # Check how long it took for algorithm to finish
    end_time = time.time()
    print("Execution time for own version: ", "--- %s seconds ---" % (end_time - start_time))

    # Create subplots of main steps
    ftc.showSteps(stepsList)

elif run_own_functions == "n":

    print("Running program with openCV functions")

    # Check the runtime
    start_time = time.time()

    # Create list with images showing the different main steps of the algorithm
    stepsList = []
    showFish = 0

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    checkerboard_dimensions, paths, clahe, cali_pa = yamlL.setup_parameters(yaml_data)

    # Load Fish images
    images, img_list_fish = ftc.loadImages(paths[0][1], edit_images=False, show_img=False)
    stepsList.append(images[showFish])

    # Load checkerboard images
    img_cali, names_cali = ftc.loadImages(paths[1][1], edit_images=False, show_img=False)

    # Calibrate camera and undistort images
    fish_cali = ftc.checkerboard_calibrateOPENCV(checkerboard_dimensions, images, img_cali,
                                                 show_img=False, recalibrate=False)
    stepsList.append(fish_cali[showFish])

    # Crop to ROI
    cropped_images = eip.crop(fish_cali, 710, 270, 850, 2600)
    stepsList.append(cropped_images[showFish])

    # Apply CLAHE
    CLAHE = claheHSL(cropped_images, 2, (25, 25))
    stepsList.append(CLAHE[showFish])

    # Threshold to create a mask for each image
    mask_cod, segmented_images = ftc.segment_codOPENCV(cropped_images)
    stepsList.append(segmented_images[showFish])
    mask_cod_CLAHE, img_segmented_cod_CLAHE = ftc.segment_cod_CLAHEOPENCV(CLAHE)

    # Wound detection
    mask_woundspots, marked_woundspots_imgs, wounds, \
    percSpotCoverage = ftc.detect_woundspotsOPENCV(segmented_images, mask_cod)
    stepsList.append(wounds[showFish])
    stepsList.append(marked_woundspots_imgs[showFish])

    # Save marked wounds images in folder
    # CLAHE images
    ftc.save_imgOPENCV(segmented_images, 'fish_pics/output_images/manual_inspection', img_list_fish,
                       "_openCV_MANUAL_INSPECTION")
    # CLAHE images
    ftc.save_imgOPENCV(img_segmented_cod_CLAHE, 'fish_pics/output_images/manual_inspection_CLAHE', img_list_fish,
                       "_openCV_MANUAL_INSPECTION_CLAHE")
    # Marked images
    ftc.save_imgOPENCV(marked_woundspots_imgs, 'fish_pics/output_images/marked_images', img_list_fish, "_openCV_marked")

    # Save a .txt file with CDI (catch damage index)
    ftc.saveCDI(img_list_fish, percSpotCoverage)

    # Check how long it took for algorithm to finish
    end_time = time.time()
    print("Execution time for openCV version: ", "--- %s seconds ---" % (end_time - start_time))

    # Create subplots of main steps
    ftc.showSteps(stepsList, CLAHE=True)
