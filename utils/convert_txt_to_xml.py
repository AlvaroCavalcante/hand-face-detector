from xml.dom.minidom import Document
import os
import os.path
from PIL import Image, ImageOps

ann_path = '/home/alvaro/Downloads/autonomy_hands_and_faces/Spruyt/labels/'      
img_path = '/home/alvaro/Downloads/autonomy_hands_and_faces/Spruyt/images/'   
xml_path = '/home/alvaro/Downloads/autonomy_hands_and_faces/Spruyt/labels_xml/'

if not os.path.exists(xml_path):
    os.mkdir(xml_path)


def writeXml(tmp, imgname, w, h, objbud, wxml):
    doc = Document()
    #Define class number according to the  classes in the .txt file
    dict = {'0': "hand",
            '1': "face"}
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)

    try:
        split_folder = wxml.split('/')
        folder_txt = doc.createTextNode(split_folder[len(split_folder)-3])
    except:
        folder_txt = doc.createTextNode("Unkown")

    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("PASCAL VOC2005")
    annotation_new.appendChild(annotation_new_txt)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode("3")
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for i in range(0, int(len(objbud))):
        objbuds=objbud[i].split(' ')
        #print(objbuds)
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(dict[objbuds[0]])
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)
        
        x1 = float(objbuds[1])
        y1 = float(objbuds[2])
        w1 = float(objbuds[3])
        h1 = float(objbuds[4])
               
        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
                
        xmin_txt2 = int((x1*w) - (w1*w)/2.0)
        #print(xmin_txt2)
        xmin_txt = doc.createTextNode(str(xmin_txt2))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt2 = int((y1*h)-(h1*h)/2.0)
        #print(ymin_txt2)
        ymin_txt = doc.createTextNode(str(ymin_txt2))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax) 
        
        xmax_txt2 = int((x1*w)+(w1*w)/2.0)
        xmax_txt2 = xmax_txt2 if xmax_txt2 <= w else w
        
        #print(xmax_txt2)
        xmax_txt = doc.createTextNode(str(xmax_txt2))
        xmax.appendChild(xmax_txt)


        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)

        ymax_txt2 = int((y1*h)+(h1*h)/2.0)
        ymax_txt2 = ymax_txt2 if ymax_txt2 <= h else h

        #print(ymax_txt2)
        ymax_txt = doc.createTextNode(str(ymax_txt2))
        ymax.appendChild(ymax_txt)

        
        # threee-1#
        # threee#
        #xmin_txts = str(int((xmin_txt*w) + (xmax_txt*w)/2.0))
        #ymax_txts = str(int((ymin_txt*h)+(ymax_txt*h)/2.0))
        #ymin_txts = str(int((ymin_txt*h)+(ymax_txt*h)/2.0))
        
        #xmin.appendChild(xmin_txts)
        #ymin.appendChild(ymin_txts)
        #xmax.appendChild(xmax_txts)
        #ymax.appendChild(ymax_txts)

                
    tempfile = tmp + "test.xml"
    with open(tempfile, "w") as f:
        f.write(doc.toprettyxml(indent='\t'))

    rewrite = open(tempfile, "r")
    lines = rewrite.read().split('\n')
    newlines = lines[1:len(lines) - 1]

    fw = open(wxml, "w")
    for i in range(0, len(newlines)):
        fw.write(newlines[i] + '\n')

    fw.close()
    rewrite.close()
    os.remove(tempfile)
    return


for files in os.walk(ann_path):
    temp = './temp/'
    if not os.path.exists(temp):
        os.mkdir(temp)
    for file in files[2]:
        print (file + "-->start!")
        img_name = os.path.splitext(file)[0] + '.jpg'
        fileimgpath = img_path + img_name
        im = ImageOps.exif_transpose(Image.open(fileimgpath))

        width = int(im.size[0])
        height = int(im.size[1])

        filelabel = open(ann_path + file, "r")
        lines = filelabel.read().split('\n')
        obj = lines[:len(lines)-1]
        #print(obj)

        filename = xml_path + os.path.splitext(file)[0] + '.xml'
        writeXml(temp, img_name, width, height, obj, filename)
    os.rmdir(temp)
