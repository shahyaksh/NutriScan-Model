
# NutriScan Model Training

This repository contains the code for training a TensorFlow-based object detection model for recognizing nutrition tables and ingredients on food packaging using the EfficientDet architecture. The dataset is managed through the Roboflow platform, and TensorFlow's Object Detection API is used for training the model. The image dataset was collected and annotated manually by us and than used for training purpose.

## Prerequisites

### Clone TensorFlow Models Repository

Ensure that you have cloned the TensorFlow models repository if it is not already present in your working directory:

```bash
git clone --depth 1 https://github.com/tensorflow/models
```

### Install Required Packages

Install the necessary packages including TensorFlow, Roboflow, and Object Detection API:

```bash
pip install tensorflow==2.12.0
pip install roboflow
```

Make sure to install the Object Detection API by navigating to the `models/research` directory:

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### Dataset Download

This project uses a dataset hosted on Roboflow. To access the dataset, make sure to provide your Roboflow API key:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("safe-bite").project("safe-bite-2")
dataset = project.version(1).download("tfrecord")
```

The downloaded dataset will be used in `.tfrecord` format for training and testing.

### Model Configuration

The script supports different models from the TensorFlow 2 Object Detection Zoo. By default, it uses the EfficientDet-D0 model for object detection. You can change the model by modifying the `MODELS_CONFIG` dictionary in the script.

Available models:

- EfficientDet-D0 (default)
- EfficientDet-D1
- EfficientDet-D2
- EfficientDet-D3

Adjust the `chosen_model` variable to use the appropriate model based on your requirements.

### Training Parameters

- `num_steps`: The total number of training steps (default: 25,000).
- `num_eval_steps`: The number of evaluation steps (default: 400).
- `batch_size`: The batch size for training (varies with the model).

### Running the Training Script

Once all configurations are set, run the following command to start the training:

```bash
python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/content/models/research/deploy/pipeline_file.config \
    --model_dir=/content/training/ \
    --alsologtostderr \
    --num_train_steps=25000 \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps=400
```

### Exporting the Model

After training, you can export the fine-tuned model by running the following command:

```bash
python models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir /content/training \
    --output_directory /content/fine_tuned_model \
    --pipeline_config_path /content/models/research/deploy/pipeline_file.config
```

The exported model will be saved in the `fine_tuned_model` directory.

### Saving Checkpoints

To save the training checkpoints to Google Drive:

```bash
cp -av /content/training /content/drive/MyDrive/Project-1/TrainingCheckpoints
```

### TensorBoard

You can monitor the training process using TensorBoard:

```bash
%tensorboard --logdir '/content/training/train'
```
# Final Model Checkpoints
You can find the final model checkpoints for direct use [here](https://drive.google.com/drive/folders/1-UWR3I01jxbpb3NX1Hd1h2EBfn99FVj1?usp=drive_link)

# Research Publication

[Delving Deep Into Nutriscan: Automated Nutrition Table Extraction and Ingredient Recognition](https://www.ijraset.com/best-journal/delving-deep-into-nutriscan-automated-nutrition-table-extraction-and-ingredient-recognition)
