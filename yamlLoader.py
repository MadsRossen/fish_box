import yaml


def yaml_loader(filepath):
    """
    Loads YAML data from a file.

    :param filepath: The path to the YAML file
    :return: YAML data
    """

    print("Loading the YAML file...")

    with open(filepath, 'r') as file_descriptor:
        yaml_data = yaml.load(file_descriptor, Loader=yaml.FullLoader)

    print("Done loading the file!")

    return yaml_data


def setup_parameters(data_yaml, printParameters=False):
    """
    Setups the parameters for the software.

    :param data_yaml: The loaded YAML data
    :return: YAML data in either string or integers
    """

    print("Setting up the parameters...")

    kernels = []
    checkerboard_dimensions = []
    paths = []
    clahe = []

    # Get the different parameters
    kernels_data = data_yaml.get('kernels')
    for open_kernel, closed_kernel in kernels_data.items():
        kernels_c = [open_kernel, closed_kernel]
        kernels.append(kernels_c)

    checkerboard_dimensions_data = data_yaml.get('checkerboard_dimensions')
    for x, y in checkerboard_dimensions_data.items():
        checkerboard_dimensions_c = [x, y]
        checkerboard_dimensions.append(checkerboard_dimensions_c)

    paths_data = data_yaml.get('paths')
    for fish, checkerboard_calibration in paths_data.items():
        paths_c = [fish, checkerboard_calibration]
        paths.append(paths_c)

    clahe_data = data_yaml.get('clahe')
    for clipLimit, tileGridSize in clahe_data.items():
        clahe_c = [clipLimit, tileGridSize]
        clahe.append(clahe_c)

    if printParameters:
        print(kernels)
        print(clahe)
        print(paths)
        print(checkerboard_dimensions)

    print("Done setting up the parameters!")

    return kernels, checkerboard_dimensions, paths, clahe