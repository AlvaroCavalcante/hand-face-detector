# Hand and Face Detector for Sign Language

Detecting the hands and the face is an important task for sign language, once those channels have most part of the information used to discriminate signs.

Based on this, this repository contains all the documentation that supports the project of a large-scale object detector for hands and faces. Besides the source code, you will find the pretrained models, datasets, and other useful resources for this context.

Although this object detection model and dataset could be used for other problems, the main application was done for sign language. That's why the dataset used to train those models were based on people executing signs. 

## Sign Language Hand and Face Dataset
Our large-scale hand and face dataset for sign language was based on the [AUTSL](https://chalearnlap.cvc.uab.cat/dataset/40/description/) dataset, which contains 43 different people performing signs, 20 backgrounds, and more than 36,000 videos.

To create the dataset annotations, we first trained a hand and face detector model using the dataset described above. After that, we used an [auto-annotation tool](https://github.com/AlvaroCavalcante/auto_annotate) to generate the bounding boxes in XML format. Finally, we manually revised all the images and the respective boxes to fix the mistakes made by the model. The generated dataset has the following statistics:

| Frames  | Hands  | Faces  |
|---|---|---|
| 477,480  |  954,960 | 477,480

**NOTE:** We tried to detect 16 frames by video, but in some cases the model was not able to detect the desired objects.

![Image](/assets/hand_face_example.png "Annotated dataset")

### Dataset split
The dataset was splitted into train, test and validation, in a proportion of 80/10/10, respectively. To split the data, the first step is to create a folder named **train**, **test** and **validation**, and then move the images and annotations into these folders. To simplify the process, we used the **dataset_split_autsl.py** script, running the following command:
```
python3 dataset_split_autsl.py
``` 
**NOTE:** It's necessary to change the images path. 

### TFRecord creation
Finally, it's necessary to convert the images and XML files into the TFRecord format to ensure TensorFlow compatibility. This can be done by following the same procedure showed before, creating a csv file and then the binary files with the commands bellow:

```
python3 utils/xml_to_csv.py -i /xml-input-path -o /csv-output-path
```
```
python3 utils/generate_tfrecord.py --csv_input=/path-to-csv --output_path ./output.record --img_path=/path-to-images --label_map=/path-to-label_map.pbtxt --n_splits n_files_to_generate 
```
In this case, we created 15 TFRecord files for test and validation sets, and 110 files for training. Every file has around 200 MB.

### Yolo annotations
To train the Yolo model, it's necessary to convert the XML annotations to TXT format. We can do this by running the following command:
```
python3 utils/xml2yolo.py
```
**NOTE:** Before running the script, be sure to change the labels, images and output paths in the beginning of the script.

After that, you can visualize the yolo bounding boxes in the images by running the following command:
```
python3 utils/show_yolo_img.py
```
> Both scripts were obtained from [this](https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5) tutorial.

## **Training the Model**
To train the object detector, the first step is to execute the model setup, by running the following script:

```
python3 hand_face_detector_setup.py --label_map_path /label_map.pbtxt --batch_size 50 --model_name "Your model name" --download_url http://tf-url.com
```

Where the model name and download url is taken from [TF Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The setup script basically configures the pipeline file to prepare the model to be trained.

After the model setup, you just need to run the following code to start the detector training:
```
python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --checkpoint_every_n=1000 \
    --num_eval_steps={num_eval_steps}
```
The model training was conducted on Google Colab. The notebook can be found [here](https://colab.research.google.com/drive/1209hYjuj449H-H_jfXLMdvnSgHYWgsq0?usp=sharing).

## **Evaluating model performance**
To evaluate the model performance during the training step, you just need to run the following command:

```
python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --eval_on_train_data=True \
    --checkpoint_dir={model_dir} \
```

When specifying the **checkpoint_dir** parameter, the last checkpoint will be used to evaluate the model performance on eval data.

## Exporting trained model
After training a great model, you probably would like to export it into the savedModel format to use it in the inference code. To do so, just run the following command:

```
python models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path {pipeline_fname} \
    --trained_checkpoint_dir {model_dir} \
    --output_directory {output_path}
```

## Testing Model Results
To verify the model working, we need to run the **hand_face_detection.py** script, with the following arguments:

- **saved_model_path**: Path of the saved_model
- **source_path**: Path of the video file to test the model. The default behavior is to use the webcam stream.
- **label_map_path**: Path of the label map used by the model.
- **compute_features**: If True, the trigronometrical features are calculated. 
- **use_docker**: Removes result visualization when running inside docker container.
- **single_person**: Define a limit of hands and face to detect considering one single person.

The **asl_bench.mp4** video, inside utils/test_videos folder is used as a commom benchmarking for the trained models, where the goal is to verify the mean FPS.

## Running Project on Nvidia Docker
An easy way to use your GPU is through the Nvidia Docker images. To do so, you can follow [this](https://www.tensorflow.org/install/docker?hl=pt-br) simple documentation. The first step is to install the nvidia support to docker, which can be done following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) tutorial.

You'll need to setup the repo into the GPG key:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
After that, just run:
```
sudo apt update
sudo apt-get install -y nvidia-container-toolkit
```
You can test if it worked by using this command:
```
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
Finally, you'll be able to run this project using your docker. To do so, first, build a new container using the Dockerfile in the project root:
```
sudo docker build . -t tf-gpu --network=host
```
**NOTE:** use --network to able internet connection.

To finish, you just need to run the following command:
```
sudo docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tf-gpu python hand_face_detection.py --source_path ./utils/test_videos/asl_bench.mp4
```

This command need to be executed inside this repository, once you'll copy all the files in your working directory into the docker container, and run itself overthere! 