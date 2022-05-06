import os
import random 
import shutil

base_path = 'autonomy_hands_and_faces/'

def move_sequence(files, base_path, train=True, labels=False):
    dest_path = 'autonomy_hands_and_faces/full_data/'
    dest_path = dest_path + 'train/' if train else dest_path+'test/'

    for f in files:
        if labels == True:
            f = f.split('.')[0]
            f = f+'.xml'
        shutil.copy(base_path+f, dest_path)
    
for folder in os.listdir(base_path):
    print(folder)
    if folder == 'full_data':
        continue

    imgs = os.listdir(base_path+folder+'/images')
    n_imgs = len(imgs)

    n_imgs_train = (n_imgs * 80) // 100

    random.Random(2).shuffle(imgs)

    train = imgs[0:n_imgs_train]
    test = imgs[n_imgs_train:]

    print('moving images')
    move_sequence(train, base_path+folder+'/images/')
    move_sequence(test, base_path+folder+'/images/', False)

    print('moving annotations')
    move_sequence(train, base_path+folder+'/labels_xml/', labels=True)
    move_sequence(test, base_path+folder+'/labels_xml/', False, labels=True)
