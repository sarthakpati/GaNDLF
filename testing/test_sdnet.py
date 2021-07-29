from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil

from GANDLF.utils import *
from GANDLF.preprocessing import *
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager

device = "cpu"
## global defines
# pre-defined segmentation model types for testing
all_models_segmentation = [
    "sdnet"
]
all_schedulers = [
    "triangle",
    "triangle_modified",
    "exp",
    "step",
    "reduce-on-plateau",
    "cosineannealing",
    "triangular",
    "triangular2",
    "exp_range",
]
all_clip_modes = ["norm", "value", "agc"]
all_norm_type = ["batch", "instance"]

patch_size = {"2D": [224, 224, 1]}

baseConfigDir = os.path.abspath(os.path.normpath("./samples"))
testingDir = os.path.abspath(os.path.normpath("./testing"))
inputDir = os.path.abspath(os.path.normpath("./testing/data"))
outputDir = os.path.abspath(os.path.normpath("./testing/data_output"))
Path(outputDir).mkdir(parents=True, exist_ok=True)


"""
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[x] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
  [x] separate tests for 2D and 3D segmentation
  [x] read default parameters from yaml config
  [x] for each type, iterate through all available segmentation model archs
  [x] call training manager with default parameters + current segmentation model arch
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
"""


def test_download_data():
    """
    This function downloads the sample data, which is the first step towards getting everything ready
    """
    urlToDownload = "https://github.com/CBICA/GaNDLF/raw/master/testing/data.zip"
    # do not download data again
    if not Path(
        os.getcwd() + "/testing/data/test/3d_rad_segmentation/001/image.nii.gz"
    ).exists():
        print("Downloading and extracting sample data")
        r = requests.get(urlToDownload)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./testing")


def test_constructTrainingCSV():
    """
    This function constructs training csv
    """
    # inputDir = os.path.normpath('./testing/data')
    # delete previous csv files
    files = os.listdir(inputDir)
    for item in files:
        if item.endswith(".csv"):
            os.remove(os.path.join(inputDir, item))

    for application_data in os.listdir(inputDir):
        currentApplicationDir = os.path.join(inputDir, application_data)

        if "2d_rad_segmentation" in application_data:
            channelsID = "image.png"
            labelID = "mask.png"
        elif "3d_rad_segmentation" in application_data:
            channelsID = "image"
            labelID = "mask"
        writeTrainingCSV(
            currentApplicationDir,
            channelsID,
            labelID,
            inputDir + "/train_" + application_data + ".csv",
        )

        # write regression and classification files
        application_data_regression = application_data.replace(
            "segmentation", "regression"
        )
        application_data_classification = application_data.replace(
            "segmentation", "classification"
        )
        with open(
            inputDir + "/train_" + application_data + ".csv", "r"
        ) as read_f, open(
            inputDir + "/train_" + application_data_regression + ".csv", "w", newline=""
        ) as write_reg, open(
            inputDir + "/train_" + application_data_classification + ".csv",
            "w",
            newline="",
        ) as write_class:
            csv_reader = csv.reader(read_f)
            csv_writer_1 = csv.writer(write_reg)
            csv_writer_2 = csv.writer(write_class)
            i = 0
            for row in csv_reader:
                if i == 0:
                    row.append("ValueToPredict")
                    csv_writer_2.writerow(row)
                    # row.append('ValueToPredict_2')
                    csv_writer_1.writerow(row)
                else:
                    row_regression = copy.deepcopy(row)
                    row_classification = copy.deepcopy(row)
                    row_regression.append(str(random.uniform(0, 1)))
                    # row_regression.append(str(random.uniform(0, 1)))
                    row_classification.append(str(random.randint(0, 2)))
                    csv_writer_1.writerow(row_regression)
                    csv_writer_2.writerow(row_classification)
                i += 1


def test_train_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 1
    # read and initialize parameters for specific data dimension
    for model in all_models_segmentation:
        parameters["model"]["architecture"] = model
        print(parameters)
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")

def test_metrics_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 1
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters["model"]["architecture"] = "sdnet"
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    shutil.rmtree(outputDir)  # overwrite previous results

    print("passed")

def test_losses_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests for losses")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 1
    parameters["model"]["architecture"] = "sdnet"
    parameters["metrics"] = ["dice"]
    # loop through selected models and train for single epoch
    for loss_type in ["dc"]:
        parameters["loss_function"] = loss_type
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )
        shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")

# test_download_data()
# test_constructTrainingCSV()
# test_train_segmentation_rad_2d(device)
test_metrics_segmentation_rad_2d('cpu')