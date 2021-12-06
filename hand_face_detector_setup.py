import os 
import urllib.request
import tarfile
import shutil
import argparse
import re
from object_detection.utils import label_map_util

def download_pretrained_model(model, download_url):
    MODEL_FILE = '_'.join(model.lower().split(' ')) + '.tar.gz'
    DEST_DIR = 'pretrained_model/'

    if not (os.path.exists(MODEL_FILE)):
        print('DOWNLOADING BASE MODEL')
        urllib.request.urlretrieve(download_url, MODEL_FILE)

    if (os.path.exists(DEST_DIR)):
        shutil.rmtree(DEST_DIR)

    os.mkdir(DEST_DIR)

    tar = tarfile.open(MODEL_FILE)
    tar.extractall(path=DEST_DIR)
    tar.close()

    os.remove(MODEL_FILE)

def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

def change_pipeline_config_values(**kwargs):
    with open(kwargs.get('pipeline_fname')) as f:
        s = f.read()
    with open(kwargs.get('pipeline_fname'), 'w') as f:
    
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(kwargs.get('fine_tune_checkpoint')), s)
        
        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train2017)(.*?")', 'input_path: "{}"'.format(kwargs.get('train_record_fname')), s)
        s = re.sub(
            '(input_path: ".*?)(val2017)(.*?")', 'input_path: "{}"'.format(kwargs.get('test_record_fname')), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(kwargs.get('label_map_pbtxt_fname')), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(kwargs.get('batch_size')), s)

        # Changing type to detection
        s = re.sub('fine_tune_checkpoint_type: ".*?"', 'fine_tune_checkpoint_type: "detection"', s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(kwargs.get('num_steps')), s)
        
        s = re.sub('total_steps: [0-9]+',
                'total_steps: {}'.format(kwargs.get('num_steps')), s)
            
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(kwargs.get('n_classes')), s)
        f.write(s)
        print(s)

def main(args):
    num_steps = args.n_steps
    num_eval_steps = args.n_eval_steps
    batch_size = args.batch_size

    MODEL = args.model_name
    DOWNLOAD_URL = args.download_url
    download_pretrained_model(MODEL, DOWNLOAD_URL)

    use_coco_checkpoint = args.use_coco_checkpoint
    if use_coco_checkpoint:
        fine_tune_checkpoint = os.getcwd() + '/pretrained_model/' + os.listdir('pretrained_model')[0] + '/checkpoint/ckpt-0'
        pipeline_fname = os.getcwd() + '/pretrained_model/' + os.listdir('pretrained_model')[0] + '/pipeline.config'
    else:
        fine_tune_checkpoint = args.fine_tune_path
        pipeline_fname = args.pipeline_file

    train_record_fname = args.train_record_path
    test_record_fname = args.test_record_path

    n_classes = get_num_classes(args.label_map_path)
    change_pipeline_config_values(num_steps=num_steps, num_eval_steps=num_eval_steps, batch_size=batch_size,
                                fine_tune_checkpoint=fine_tune_checkpoint, pipeline_fname=pipeline_fname, n_classes=n_classes,
                                train_record_fname=train_record_fname, test_record_fname=test_record_fname, label_map_pbtxt_fname=args.label_map_path)
    
    print('Model Setup completed!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_map_path', type=str, default='/home/alvaro/Documentos/body-detection/utils/label_map.pbtxt')
    parser.add_argument('--n_steps', type=int, default=4000)
    parser.add_argument('--n_eval_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--download_url', type=str)
    parser.add_argument('--train_record_path', type=str)
    parser.add_argument('--test_record_path', type=str)
    parser.add_argument('--use_coco_checkpoint', type=bool, default=True)
    args = parser.parse_args()

    main(args)