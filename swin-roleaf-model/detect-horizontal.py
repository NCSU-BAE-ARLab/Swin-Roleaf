# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py 
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import os.path as osp
import csv
import operator
import math
import xml.etree.ElementTree as ET

IMGSIZE_W, IMGSIZE_H = 1024, 1024

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly


@torch.no_grad()
def run(weights=ROOT / 'train-model/best.pt',  # model.pt path(s)
        source=ROOT / 'dataset-corn',  # file/dir/URL/glob, 0 for webcam
        imgsz=(1024, 1024),  # inference size (height, width)
        conf_thres=0.9,  # confidence threshold
        iou_thres=0.9,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'detect-results'  ,  # save results to project/name
        name='results',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale polys from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *poly, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        poly = torch.tensor(poly).view(1,8).view(-1).tolist()
                        line = (cls, *poly, conf) if save_conf else (cls, *poly)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add poly to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.poly_label(poly, label, color=colors(c, True))
                        if save_crop: # Swin-Roleaf doesn't support it yet
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            pass

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('*.txt')))} labels saved to {save_dir}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    return save_dir


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'train-model/best.pt')
    parser.add_argument('--source', type=str, default='dataset-corn', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1024], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='detect-results', help='save results to project/name')
    parser.add_argument('--name', default='results', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

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

def run_covert(read_dir):
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
                    angle_csv = 90 - round(abs(a / math.pi * 180 - 90), 0)
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
    final_csv_file = "final_output.csv"
    merge_csv_files(read_dir, final_csv_file)

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    save_dir = run(**vars(opt))
    run_covert(save_dir)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
