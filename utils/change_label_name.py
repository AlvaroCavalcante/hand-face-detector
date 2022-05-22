import glob

base_path = '/home/alvaro/Documentos/video2tfrecord/validation/'

xml_files = glob.glob(base_path+'*.xml')

print('Starting label name changing')

for xml in xml_files:
    initial_xml = open(xml, "r")

    lines_of_file = initial_xml.readlines()

    if len(lines_of_file) > 1:
        new_line = []
        for i, line in enumerate(lines_of_file):
            if 'hand_1' in line:
                line_content = line.replace('hand_1', 'hand')
            else:
                line_content = line.replace('hand_2', 'hand')
            new_line.append(line_content)
        lines_of_file = new_line
    else:
        lines_of_file[0] = lines_of_file[0].replace('hand_1', 'hand')
        lines_of_file[0] = lines_of_file[0].replace('hand_2', 'hand')

    out_xml = open(xml, "w")
    out_xml.writelines(lines_of_file)
    out_xml.close()

print('Process finished')
