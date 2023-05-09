This folder could contain TF Models to test the execution of the script. 

To convert the models checkpoint to saved models, run the following script:

```
python3 models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path /home/alvaro/Desktop/hand-face-detector/utils/pipeline_resnet_101.config \
    --trained_checkpoint_dir /home/alvaro/Desktop/hand-face-detector/utils/models/resnet_101_100p \
    --output_directory /home/alvaro/Desktop/hand-face-detector/utils/models/
```