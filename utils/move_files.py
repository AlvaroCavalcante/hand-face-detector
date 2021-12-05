import os 
import shutil

base_path = '/home/alvaro/Downloads/autonomy_hands_and_faces/Autonomy/labels/'
folders = os.listdir(base_path)

for folder in folders:
    if folder != 'labels':
        for file in os.listdir(base_path+folder):
            try:
                shutil.move(base_path+folder+'/'+file, base_path+'labels/')
            except Exception as e:
                print(e)
                print(folder, file)