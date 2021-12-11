# Multi-Cue detector for sign language

This repo contains the code used to train an object detection algorithm to detect human faces and hands as a preprocessing step to a sign language recognition system

## Hand Dataset
The initial dataset used to train the hand detector was the [hand dataset](https://www.robots.ox.ac.uk/~vgg/data/hands/).

The dataset labels are in the .mat format and need to be converted into XML in order to train the detector using TensorFlow. To do so we used the ```mat_to_xml.py``` algorithm. To run the code, just use:

```
python3 utils/mat_to_xml.py --img-path /imgs-folder-path --annotations-path /mat-annotations-folder --output-path /folder-to-store-xml
```

After transforming the annotations into XML format, it's necessary to convert then into CSV. This can be easily done by the following command:
```
python3 utils/xml_to_csv.py -i /xml-path -o /csv-output-path
```
It's possible that some labels were incorrectly converted in this process. In order to check if the train and test labels are correct you can run the **verify_labels.py** script, changing the base path to indicate your current path.

Finally, it's necessary to convert the CSV files into TFRecord format to train the object detector using the TF Data API. To do so, just run the following code:
```
python3 utils/generate_tfrecord.py --csv_input=/path-to-csv --output_path ./output.record --img_path=/path-to-images --label_map=./utils/label_map.pbtxt --n_splits n_files_to_generate 
```

## Hand and Face Dataset
Once the goal of the most recent work is to detect hand and face simultaneously, another dataset was required, since it's necessary to have hands and faces annotated at the same time. For this pourpose, we used the "Autonomy" dataset, which contains a collection of other open source datasets but with both hand and face annotated. The original paper can be found [here](https://autonomy.cs.sfu.ca/doc/mohaimenian_iros2018.pdf).

The annotations are present in TXT format (used by default in Yolo), and could be converted to XML by using the **convert_txt_to_xml.py** script. To run the code, first it's necessary to change the dataset path in the beginning of the script and the label map with the class annotations. After that, just run the following command.
```
python3 utils/convert_txt_to_xml.py
```
This code was got from [this](https://github.com/MuhammadAsadJaved/Important-shells) open-source repo.

**Note:** the "Autonomy" folder of this dataset is structured in multiple folders. To convert the annotations, first I used the script entitled "move_files.py" to centralize all the images and annotations.

After that, it's also necessary to convert the XML annotations and the images into TFRecords. To do so, first we need to centralize all the data in the train and test folders. It can be done by running the following code:
```
python3 utils/dataset_split.py
```
The default division proportion is 80/20, respectivelly. To finish the dataset preparation, it's necessary to transform the XML annotations into CSV and then in TFRecords. This can be done following the same procedure showed before.

### **Training the model**
To train the object detector, the first step is to execute the model setup, by running the following script:

```
python3 hand_face_detector_setup.py --label_map_path /label_map.pbtxt --batch_size 50 --model_name "Your model name" --download_url http://tf-url.com
```

Where the model name and download url is taken from [TF Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The setup script basically configures the pipeline file to prepare the model to be trained.

After the model setup, you just need to run the following code to start the detector training:
```
python /models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --checkpoint_every_n=1000 \
    --num_eval_steps={num_eval_steps}
```

The model training was conducted on Google Colab. The notebook can be found [here](https://colab.research.google.com/drive/1209hYjuj449H-H_jfXLMdvnSgHYWgsq0?usp=sharing).

## Testing model results
To verify the model working, we need to run the **hand_face_detection.py** script, with the following arguments:

- **saved_model_path**: Path of the saved_model
- **source_path**: Path of the video file to test the model. The default behavior is to use the webcam stream.
- **label_map_path**: Path of the label map used by the model.
- **compute_features**: If True, the trigronometrical features are calculated. 

The **asl_bench.mp4** video, inside utils/test_videos folder is used as a commom benchmarking for the trained models, where the goal is to verify the mean FPS.