from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil, sys, yaml, torch
import SimpleITK as sitk

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame, elastic, mri_artifact
from GANDLF.utils import *
from GANDLF.preprocessing import *
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager
from GANDLF.cli.main_run import main_run
from GANDLF.cli.preprocess_and_save import preprocess_and_save

device = "cpu"
## global defines
# pre-defined segmentation model types for testing
all_models_segmentation = [
    "unet",
    "fcn",
    "uinc",
    "msdnet",
]
# pre-defined regression/classification model types for testing
all_models_regression = ["densenet121", "vgg16"]
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

patch_size = {"2D": [128, 128, 1], "3D": [32, 32, 32]}

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
    parameters["model"]["num_channels"] = 3
    # read and initialize parameters for specific data dimension
    for model in all_models_segmentation:
        parameters["model"]["architecture"] = model
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


def test_train_segmentation_rad_3d(device):
    print("Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    for model in all_models_segmentation:
        parameters["model"]["architecture"] = model
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


def test_train_regression_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check=False
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
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


def test_train_brainage_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check=False
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    parameters["model"]["architecture"] = "brain_age"
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


def test_train_regression_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check=False
    )
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
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


def test_train_classification_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check=False
    )
    parameters["modality"] = "rad"
    parameters["track_memory_usage"] = True
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
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


def test_train_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
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


def test_inference_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_scheduler_classification_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["model"]["architecture"] = "densenet121"
    # loop through selected models and train for single epoch
    for scheduler in all_schedulers:
        parameters["scheduler"] = scheduler
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    shutil.rmtree(outputDir)
    print("passed")


def test_clip_train_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["model"]["architecture"] = "vgg16"
    # loop through selected models and train for single epoch
    for clip_mode in all_clip_modes:
        parameters["clip_mode"] = clip_mode
        # shutil.rmtree(outputDir)  # overwrite previous results
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


def test_normtype_train_segmentation_rad_3d(device):
    # read and initialize parameters for specific data dimension
    print("Starting 3D Rad segmentation tests for normtype")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    for norm in ["batch", "instance"]:
        for model in ["resunet", "unet", "fcn"]:
            parameters["model"]["architecture"] = model
            parameters["model"]["norm_type"] = norm
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
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters["model"]["architecture"] = "unet"
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


def test_metrics_regression_rad_2d(device):
    print("Starting 2D Rad regression tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = {}
    parameters["metrics"]["mse"] = {}
    parameters["metrics"]["accuracy"] = {}
    parameters["metrics"]["accuracy"]["threshold"] = 0.5
    parameters["model"]["architecture"] = "vgg11"
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
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "resunet"
    parameters["metrics"] = ["dice"]
    # loop through selected models and train for single epoch
    for loss_type in ["dc", "dc_log", "dcce", "dcce_logits"]:
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


def test_config_read():
    print("Starting testing reading configuration")
    # read and parse csv
    parameters = parseConfig(
        os.path.abspath(baseConfigDir + "/config_all_options.yaml"), version_check=True
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    if not parameters:
        sys.exit(1)
    data_loader = ImagesFromDataFrame(training_data, parameters, True)
    if not data_loader:
        sys.exit(1)
    print("passed")


def test_cli_function_preprocess():
    print("Starting testing cli function preprocess")
    file_config = os.path.join(testingDir, "config_segmentation.yaml")
    file_config_temp = os.path.join(testingDir, "config_segmentation_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(file_config_temp):
        os.remove(file_config_temp)
        parameter_pickle_file = os.path.join(outputDir, "parameters.pkl")
        if os.path.exists(parameter_pickle_file):
            os.remove(parameter_pickle_file)
    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    parameters = parseConfig(file_config)
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = "[0, 255||125]"
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "unet"
    parameters["metrics"] = ["dice"]
    parameters["patch_sampler"] = "label"
    parameters["weighted_loss"] = True
    parameters["save_output"] = True

    # store this separately for preprocess testing
    with open(file_config_temp, "w") as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    preprocess_and_save(file_data, file_config_temp, outputDir)
    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_cli_function_mainrun(device):
    print("Starting testing cli function main_run")
    file_config_temp = os.path.join(testingDir, "config_segmentation_temp.yaml")
    # if preprocess wasn't run, this file should not be present
    if not os.path.exists(file_config_temp):
        file_config_temp = os.path.join(testingDir, "config_segmentation.yaml")

    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    main_run(file_data, file_config_temp, outputDir, True, device, True)
    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_preprocess_functions():
    print("Starting testing preprocessing functions")
    input_tensor = torch.rand(1, 3, 256, 256)
    input_transformed = normalize_imagenet(input_tensor)
    input_transformed = normalize_standardize(input_tensor)
    input_transformed = normalize_div_by_255(input_tensor)
    input_transformed = threshold_intensities(input_tensor, 0.25, 0.75)
    assert (
        torch.count_nonzero(input_transformed[input_transformed < 0.25] > 0.75) == 0
    ), "Input should be thresholded"

    input_transformed = clip_intensities(input_tensor, 0.25, 0.75)
    assert (
        torch.count_nonzero(input_transformed[input_transformed < 0.25] > 0.75) == 0
    ), "Input should be thresholded"

    input_transformed = tensor_rotate_90(input_tensor, (1))
    input_transformed = tensor_rotate_180(input_tensor, (1))

    non_zero_normalizer = NonZeroNormalizeOnMaskedRegion()
    input_transformed = non_zero_normalizer(input_tensor)

    elastic_generator = elastic(patch_size=[32, 32, 1])
    input_transformed = elastic_generator(input_tensor)
    elastic_generator = elastic()
    input_transformed = elastic_generator(input_tensor)
    mri_artefact_generator = mri_artifact()
    input_transformed = mri_artefact_generator(input_tensor)

    input_image = sitk.GetImageFromArray(input_tensor[0].numpy())
    img_resized = resample_image(
        input_image,
        resize_image_resolution(input_image, [128, 128]),
        interpolator=sitk.sitkNearestNeighbor,
    )
    img_tensor = get_tensor_for_dataloader(img_resized)
    assert img_tensor.shape == (1, 3, 128, 128), "Resampling should work"

    input_tensor = torch.rand(1, 256, 256, 256)
    cropper = CropExternalZeroplanes(patch_size=[128, 128, 128])
    input_transformed = cropper(input_tensor)
