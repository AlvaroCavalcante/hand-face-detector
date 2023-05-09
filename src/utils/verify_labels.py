import csv
import cv2 
import os
import numpy as np

csv_files = ['train.csv']
folders = ['train']

for i in range(len(folders)):
    FOLDER = '/home/alvaro/Downloads/autonomy_hands_and_faces/full_data/'+folders[i]
    CSV_FILE = '/home/alvaro/Downloads/autonomy_hands_and_faces/full_data/'+csv_files[i]
    erros_files = []

    with open(CSV_FILE, 'r') as fid:
        
        print('Checking file:', CSV_FILE, 'in folder:', FOLDER)
        
        file = csv.reader(fid, delimiter=',')
        first = True
        
        cnt = 0
        error_cnt = 0
        error = False
        for row in file:
            if error == True:
                error_cnt += 1
                error = False
                
            if first == True:
                first = False
                continue
            
            cnt += 1
            
            name, width, height, xmin, ymin, xmax, ymax = row[0], int(row[1]), int(row[2]), int(row[4]), int(row[5]), int(row[6]), int(row[7])
            
            path = os.path.join(FOLDER, name)
            img = cv2.imread(path)
            
            
            if type(img) == type(None):
                error = True
                print('Could not read image', img)
                continue
            
            org_height, org_width = img.shape[:2]
            
            if org_width != width:
                error = True
                print('Width mismatch for image: ', name, width, '!=', org_width)
            
            if org_height != height:
                error = True
                print('Height mismatch for image: ', name, height, '!=', org_height)
            
            if xmin > org_width:
                error = True
                print('XMIN > org_width for file', name)
                
            if xmin <= 0:
                error = True
                print('XMIN < 0 for file', name)
                
            if xmax > org_width:
                error = True
                print('XMAX > org_width for file', name)
            
            if ymin > org_height:
                error = True
                print('YMIN > org_height for file', name)
            
            if ymin <= 0:
                error = True
                print('YMIN < 0 for file', name)
            
            if ymax > org_height:
                error = True
                print('YMAX > org_height for file', name)
            
            if xmin >= xmax:
                error = True
                print('xmin >= xmax for file', name)
                
            if ymin >= ymax:
                error = True
                print('ymin >= ymax for file', name)
            
            if error == True:
                print('Error for file: %s' % name)
                erros_files.append(name)
                print()
            
        print('Files with error', erros_files)
        print('Checked %d files and realized %d errors' % (cnt, error_cnt))