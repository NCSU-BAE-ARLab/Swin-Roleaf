import os
import os.path as osp
import csv
import operator
import math
import xml.etree.ElementTree as ET

IMGSIZE_W, IMGSIZE_H = 1024, 1024

def get_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
# change the lable format
def convert_xy4_to_xywha(xy4, IMGSIZE_W, IMGSIZE_H):
    p1 = [eval(xy4[0]), eval(xy4[1])]
    p2 = [eval(xy4[2]), eval(xy4[3])]
    p3 = [eval(xy4[4]), eval(xy4[5])]
    p4 = [eval(xy4[6]), eval(xy4[7])]

    cx = sum((p1[0], p2[0], p3[0], p4[0])) / 4
    cy = sum((p1[1], p2[1], p3[1], p4[1])) / 4

    distances = list()
    distances.append(get_distance(p1, p2))
    distances.append(get_distance(p1, p3))
    distances.append(get_distance(p1, p4))
    distances.append(get_distance(p2, p3))
    distances.append(get_distance(p2, p4))
    distances.append(get_distance(p3, p4))
    distances.sort()

    w = (distances[2] + distances[3]) / 2
    h = (distances[0] + distances[1]) / 2

    pp1, pp2, pp3, pp4 = sorted([p1, p2, p3, p4], key=operator.itemgetter(1))
    pp0 = pp2
    d = get_distance(pp1, pp0)
    temp = abs(d - w)
    for ppi in [pp3, pp4]:
        d = get_distance(pp1, ppi)
        if abs(d - w) < temp:
            temp = abs(d - w)
            pp0 = ppi

    dy = pp0[1] - pp1[1]
    dx = pp0[0] - pp1[0]

    if dy < 1e-6:
        angle = 0
    elif abs(dx) < 1e-6:
        angle = 90
    else:
        angle = int(math.atan(dy / dx) * 180 / math.pi)

    if angle < 0:
        angle += 180

    angle = angle * math.pi / 180.0
    return cx, cy, w, h, angle

def change_extension(file_name, new_extension):
    base = os.path.splitext(file_name)[0]
    return f"{base}.{new_extension}"

def merge_csv_files(read_dir, final_csv_filename):
    # Ensure the final CSV is not in the same directory as the source CSVs
    final_csv_path = osp.join(read_dir, final_csv_filename)
    with open(final_csv_path, 'w', newline='') as f_out:
        csv_writer = csv.writer(f_out)
        for file in os.listdir(read_dir):
            file_path = osp.join(read_dir, file)
            # Skip the final CSV file while merging
            if file.endswith(".csv") and file_path != final_csv_path:
                with open(file_path, 'r') as f_in:
                    csv_reader = csv.reader(f_in)
                    for row in csv_reader:
                        csv_writer.writerow(row)

def run(read_dir):
    for file in os.listdir(read_dir):
        if file.endswith(".txt"):
            read_txt = osp.join(read_dir, file)
            write_csv = osp.join(read_dir, change_extension(file, "csv"))
            write_xml = osp.join(read_dir, change_extension(file, "xml"))

            annotation = ET.Element('annotation')
            folder = ET.SubElement(annotation, 'folder')
            folder.text = 'exp18'
            filename_xml = ET.SubElement(annotation, 'filename')
            filename_xml.text = '0091'
            path = ET.SubElement(annotation, 'path')
            path.text = 'C:/Users/whe/Downloads/exp18'

            source = ET.SubElement(annotation, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'database'

            size = ET.SubElement(annotation, 'size')
            width = ET.SubElement(size, 'width')
            width.text = '512'
            height = ET.SubElement(size, 'height')
            height.text = '455'
            depth = ET.SubElement(size, 'depth')
            depth.text = '3'

            segmented = ET.SubElement(annotation, 'segmented')
            segmented.text = '0'

            with open(read_txt, 'r') as fp, open(write_csv, 'w', newline='') as csv_file:
                lines = fp.readlines()
                csv_writer = csv.writer(csv_file)

                for line in lines:
                    cls_name, *xy4 = line.strip().split()

                    x, y, w, h, a = convert_xy4_to_xywha(xy4, IMGSIZE_W, IMGSIZE_H)
                    cls = 0
                    angle_csv = round(abs(a / math.pi * 180 - 90), 0)
                    csv_writer.writerow([angle_csv])

                    object_ = ET.SubElement(annotation, 'object')
                    type = ET.SubElement(object_, 'type')
                    type.text = 'robndbox'
                    name = ET.SubElement(object_, 'name')
                    name.text = 'Leaf Azimuth Angle: '+str(angle_csv) + '°'
                    pose = ET.SubElement(object_, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(object_, 'truncated')
                    truncated.text = '0'
                    difficult = ET.SubElement(object_, 'difficult')
                    difficult.text = '0'

                    robndbox = ET.SubElement(object_, 'robndbox')
                    cx_ = ET.SubElement(robndbox, 'cx')
                    cx_.text = str(x)
                    cy_ = ET.SubElement(robndbox, 'cy')
                    cy_.text = str(y)
                    w_ = ET.SubElement(robndbox, 'w')
                    w_.text = str(w)
                    h_ = ET.SubElement(robndbox, 'h')
                    h_.text = str(h)
                    angle_ = ET.SubElement(robndbox, 'angle')
                    angle_.text = str(a)

            tree = ET.ElementTree(annotation)
            tree.write(write_xml)

    # Merge all CSV files into one final CSV file
    final_csv_file = osp.join(read_dir, "final_output.csv")
    merge_csv_files(read_dir, final_csv_file)

# Outside of run function
if __name__ == '__main__':
    read_dir = r'..\swin-roleaf-model\detect-results\labels'
    final_csv_filename = "output_angle.csv"
    run(read_dir)
    merge_csv_files(read_dir, final_csv_filename)