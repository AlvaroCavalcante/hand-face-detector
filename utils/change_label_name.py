import glob

base_path = '/home/alvaro/Documentos/video2tfrecord/test/'

xml_files = glob.glob(base_path+'*.xml')

for xml in xml_files:
    initial_xml = open(xml, "r")

    lines_of_file = initial_xml.readlines()

    lines_of_file[0] = lines_of_file[0].replace('hand_1', 'hand')
    lines_of_file[0] = lines_of_file[0].replace('hand_2', 'hand')

    out_xml = open(xml, "w")
    out_xml.writelines(lines_of_file)
    out_xml.close()
