import glob
import random 
import shutil

base_path = '/home/alvaro/Documentos/video2tfrecord/'

def move_sequence(files, base_path, suffix, labels=False):
    dest_path = base_path + suffix

    for f in files:
        if labels == True:
            f = f.split('.')[0]
            f = f+'.xml'
        shutil.copy(f, dest_path)
    
imgs = glob.glob(base_path+'object_detection_db/*.jpg')
n_imgs = len(imgs)

n_imgs_train = (n_imgs * 80) // 100
n_imgs_test = (n_imgs * 10) // 100
n_imgs_val = (n_imgs * 10) // 100

random.Random(2).shuffle(imgs)

train = imgs[0:n_imgs_train]
test = imgs[n_imgs_train:n_imgs_train + n_imgs_test]
validation = imgs[n_imgs_train+n_imgs_test:]

print('moving images')
move_sequence(train, base_path, 'train/')
move_sequence(test, base_path, 'test/')
move_sequence(validation, base_path, 'validation/')

print('moving annotations')
move_sequence(train, base_path, 'train/', labels=True)
move_sequence(test, base_path, 'test/', labels=True)
move_sequence(test, base_path, 'validation/', labels=True)
