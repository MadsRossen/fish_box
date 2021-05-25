import cv2
import functions as ft
import functions_openCV as ftc
import yamlLoader as yamlL
import extremeImageProcessing as eip
import time
import sys

# Check if the user wanna use our functions or not
run_own_functions = sys.argv[1]
experimental = 0

# Does the user wanna use our own functions or OpenCV functions. y for own functions, n for openCV
if run_own_functions == 'y':

    print("Running program with own built-in functions")

    # Check the runtime
    start_time = time.time()

    # First check if the software should run the experimental
    if len(sys.argv) > 2:
        experimental = sys.argv[2]
        # debug = sys.argv[3]

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    _, paths, clahe, cali_pa = yamlL.setup_parameters(yaml_data)

    # load Fish images
    images, names = ft.loadImages(paths[0][1], False, False, 60)

    # Calibrate camera and undistort images
    fish_cali = eip.undistort(images, cali_pa[0][1], cali_pa[1][1], cali_pa[2][1], cali_pa[3][1], cali_pa[4][1],
                              cali_pa[5][1], True)

    # Crop to ROI to make the processing time smaller
    cropped_images = eip.crop(fish_cali, 710, 200, 720, 2500)

    # Save the undistorted image for further inspection
    cv2.imwrite('fishpics/UndiImg/undi_fish.JPG', cropped_images[0])

    # Threshold to create a mask for each image
    masks, segmented_images = eip.segment_cod(cropped_images, False)

    # Use morphology on images
    images_morph, res_images = ft.morphology_operations(masks, segmented_images, 5, 7, False, False)

    # Blood spot detection
    masks_woundspot, _, marked_woundspots_imgs, _, damage_percentage = ft.detect_woundspots(res_images)

    # CDI in a text file
    ftc.saveCDI(names, damage_percentage)

    # If the user wanna try the experimental features of the program
    if experimental == 'e':
        # Get the contours
        contour = ft.find_contours(masks, cropped_images, False, False)

        # Find contours middle point
        xcm, ycm = ft.contour_MOC(cropped_images, contour)

        # Raytracing
        rot_img = ft.raytracing(cropped_images, xcm, ycm, contour)
    else:
        # Display the final images
        cv2.imshow("Final segmented image", segmented_images[0])

    print("Execution time: ", "--- %s seconds ---" % (time.time() - start_time))

    # Display the final images
    cv2.imshow("Final bloodspot detection", marked_woundspots_imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif run_own_functions == "n":

    print("Running program with openCV functions")

    # Check the runtime
    start_time = time.time()

    stepsList = []
    showFish = 0

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    checkerboard_dimensions, paths, clahe, cali_pa = yamlL.setup_parameters(yaml_data)

    # Load Fish images
    images, names, img_list_fish = ftc.loadImages(paths[0][1], False, False, 40)
    stepsList.append(images[showFish])

    # Load checkerboard images
    img_cali, names_cali, _ = ftc.loadImages(paths[1][1], False, False, 40)

    # Calibrate camera and undistort images
    fish_cali = ftc.checkerboard_calibrateOPENCV(checkerboard_dimensions, images, img_cali,
                                                 show_img=False, recalibrate=False)
    stepsList.append(fish_cali[showFish])

    # Crop to ROI
    cropped_images = eip.crop(fish_cali, 710, 200, 720, 2500)
    cv2.imwrite('fish_pics/segment_this.JPG', cropped_images[0])
    stepsList.append(cropped_images[showFish])

    # Threshold to create a mask for each image
    mask_cod, img_segmented_cod = ftc.segment_codOPENCV(cropped_images)
    stepsList.append(img_segmented_cod[showFish])

    # Blood spot detection
    mask_bloodspots, bloodspots, marked_bloodspots, \
    percSpotCoverage = ftc.detect_bloodspotsOPENCV(img_segmented_cod, mask_cod)
    stepsList.append(bloodspots[showFish])
    stepsList.append(marked_bloodspots[showFish])

    # Save marked blood spots images in folder
    ftc.save_imgOPENCV(marked_bloodspots, 'fish_pics/output_images', img_list_fish)

    # Save a .txt file with CDI (catch damage index)
    ftc.saveCDI(img_list_fish, percSpotCoverage)

    # Check how long it took for algorithm to finish
    end_time = time.time()
    print('Total time for algoritm: ', end_time - start_time, 'sec')

    # Create subplots of main steps
    ftc.showSteps(stepsList)

    print("Execution time: ", "--- %s seconds ---" % (time.time() - start_time))
