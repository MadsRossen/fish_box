import cv2
import functions as ft
import yamlLoader as yamlL
import extremeImageProcessing as eip
import time
import sys

# Check if the user wanna use our functions or not
run_own_functions = sys.argv[1]

if run_own_functions == 'y':
    print("Running program with own built functions")

    # Check the runtime
    start_time = time.time()

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    kernels, checkerboard_dimensions, paths, clahe, cali_pa = yamlL.setup_parameters(yaml_data)

    # load Fish images
    images, names = ft.loadImages(paths[0][1], False, False, 40)

    # Load checkerboard images
    img_cali, names_cali = ft.loadImages(paths[1][1], False, False, 40)

    # Calibrate camera and undistort images
    # fish_cali = eip.undistort(images, cali_pa[0][1], cali_pa[1][1], cali_pa[2][1], cali_pa[3][1], cali_pa[4][1],
                             # cali_pa[5][1], True)

    # Crop to ROI
    cropped_images = eip.crop(images, 700, 450, 600, 2200)

    # Threshold to create a mask for each image
    # masks, segmented_images = eip.segment_cod(cropped_images, clahe[0][1], clahe[1][1], False)
    masks, segmented_images = eip.segment_cod(cropped_images, False)

    # Bloodspot detection
    # blodspot_img = ft.detect_bloodspots()

    # Get the contours
    contour = ft.find_contours(masks, cropped_images, False, True)

    # Find contours middle point
    xcm, ycm = ft.contour_MOC(cropped_images, contour)

    # Raytracing
    rot_img = ft.rotateImages(cropped_images, xcm, ycm, contour)

    print("Execution time for optimized item/itemset function: ", "--- %s seconds ---" % (time.time() - start_time))

    # Display the final images
    cv2.imshow("Final", rot_img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif not run_own_functions == "n":

    print("Running program with openCV functions")

    # Check the runtime
    start_time = time.time()

    # Load in yaml data from the file
    yaml_data = yamlL.yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    kernels, checkerboard_dimensions, paths, clahe = yamlL.setup_parameters(yaml_data)

    # load Fish images
    images, names = ft.loadImages(paths[0][1], False, False, 40)

    # Load checkerboard images
    img_cali, names_cali = ft.loadImages(paths[1][1], False, False, 40)

    # Calibrate camera and undistort images
    fish_cali = ft.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, False)

    # Crop to ROI
    cropped_images = eip.crop(fish_cali, 700, 450, 600, 2200)

    # Threshold to create a mask for each image
    masks, segmented_images = eip.segment_cod(cropped_images, clahe[0][1], clahe[1][1], False)

    # Bloodspot detection
    # blodspot_img = ft.detect_bloodspots()

    # Get the contours
    contour = ft.find_contours(masks, cropped_images, False, True)

    # Find contours middle point
    xcm, ycm = ft.contour_MOC(cropped_images, contour)

    # Raytracing
    rot_img = ft.rotateImages(cropped_images, xcm, ycm, contour)

    # Display the final images
    cv2.imshow("Final", rot_img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


