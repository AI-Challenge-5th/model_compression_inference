# +
import argparse
import glob
import json
import os
import sys

file_dir = os.path.dirname('./yolov5/val.py')
sys.path.append(file_dir)

from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import create_dataloader
from yolor.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from yolor.utils.loss import compute_loss
from yolor.utils.metrics import ap_per_class
from yolor.utils.plots import plot_images, output_to_target
from yolor.utils.torch_utils import select_device, time_synchronized

from yolor.models.models import *
from models.common import DetectMultiBackend

memory_limit = 6 * 1000 * 1000 * 1000 / torch.cuda.get_device_properties('cuda:0').total_memory
torch.cuda.set_per_process_memory_fraction(memory_limit)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def creat_dataset_txt():
    data_path = Path("/home/agc2021/dataset/")
#     data_path = Path("./dataset/")
    images = [str(x) for x in list(data_path.glob("t4_*.jpg"))]
    with open("agc2021_testset.txt", 'w') as f:
        for img in images:
            f.write(img+"\n")

def calc_model_param(model):
    return sum(p.numel() for p in model.parameters())

def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0,
         model_archi='yolor'):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
#         save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_dir = Path(opt.project) / opt.name # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    half = device.type != 'cpu' 
    # Load model
    if model_archi=='yolor':
        model = Darknet(opt.cfg).to(device)
        try:
            if weights[0].endswith("pth"):
                model = torch.load(weights[0], map_location=device)
            else:
                ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
                ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(ckpt['model'], strict=False)
        except Exception as e:
            print("model load fail! : ",e)
            load_darknet_weights(model, weights[0])
        
        if half:
            model.half()
        
    else:
        model = DetectMultiBackend(weights, device=device, dnn=False)
        if half:
            model.model.half()
    imgsz = check_img_size(imgsz, s=64)  # check img_size

    
    # Configure
    model.eval()
#     is_coco = data.endswith('coco.yaml')  # is COCO dataset
    is_coco = True  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    if model_archi=='yolor':
        model_params_num = calc_model_param(model)
    elif model_archi=='yolov5':
        model_params_num = calc_model_param(model.model)
    else:
        model_params_num = calc_model_param(model)
    
    jdict.append({'framework':'pytorch'})
    jdict.append({'parameters':int(model_params_num)})
    
#     for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            if model_archi=='yolor':
                inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            elif model_archi=='yolov5':
                inf_out, train_out = model(img, augment=augment, val=True)  # inference and training outputs
            else:
                inf_out, train_out = model(img)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
#             if save_txt:
#                 gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
#                 x = pred.clone()
#                 x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
#                 for *xyxy, conf, cls in x:
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                     with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
#                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
#             if plots and len(wandb_images) < log_imgs:
#                 box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
#                              "class_id": int(cls),
#                              "box_caption": "%s %.3f" % (names[cls], conf),
#                              "scores": {"class_score": conf},
#                              "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
#                 boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
#                 wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                if isinstance(path.stem, str):
                    image_string = path.stem.split('_')[1]
                    image_id = int(image_string)
                else:
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})


    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        pred_json = "answersheet_4_04_nuxlear.json"  # predictions json
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--weights', nargs='+', type=str, default=['agc2021.pt'], help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='agc2021.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='test', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='eval', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='./yolor/cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./yolor/data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('agc2021.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    
    creat_dataset_txt()
    
    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             model_archi='yolov5'
             )

    
