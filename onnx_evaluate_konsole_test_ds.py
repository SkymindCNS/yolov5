from cProfile import label
from pickle import FALSE
import onnxruntime
import torch
import cv2
import numpy as np
import time
import torchvision
from utils.plots import Annotator, colors
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
onnx_path    = "onnx_weights/best_1.onnx"
test_folder_path = "../yolov5_konsole_test_set"
half = FALSE
names = ['positive', 'negative', 'invalid-empty-result-region', 'invalid-smudge']
output_folder = "yolov5_evaluation"


conf_thres=0.9  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1
classes=None
agnostic=False
agnostic_nms=False

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def inference(session, path):
    s = f'image {path}: '
    # dataasets.py --Load Images
    img0 = cv2.imread(path)
    im = letterbox(img0, [640, 640], stride=64, auto=False)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    #inference
    im = im.cpu().numpy().astype(np.float32)  # torch to numpy
    pred = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
    pred = torch.tensor(pred)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):
        annotator = Annotator(img0, line_width=3, example=str(names))
        s += '%gx%g ' % im.shape[2:]  # print string
        xyxy = None
        class_idx = None
        predicted_class = None
        confidence = None
        if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique(): #last element is the class
                    n = (det[:, -1] == c).sum()  # detections per class
                    class_idx = int(c)
                    predicted_class = names[class_idx]
                    s += f"{n} {predicted_class}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    s += f'conf: {conf:.2f}, loc: {xyxy}'
                
                confidence = conf.item()
        
    output_result = {
        "image": path,
        "rr_loc": xyxy,
        "class": class_idx,
        "label": predicted_class,
        "conf": confidence
    }
    
    labelled_img = annotator.result()
    return output_result, labelled_img

def get_correct_label(label_file):
    with open(label_file) as f:
        correct_class = int(f.read().split(' ')[0])
    return correct_class

def get_label_from_folder(img_path):
    "return numerical label based on names index"
    splitted_path =  os.path.split(img_path)[-2]
    folder = os.path.split(splitted_path)[-1]
    y = names.index(folder)
    return y

def create_evaluation_folder(parent_img_dir, output_dir_name = "yolov5_evaluation"):
    classes = [f.stem for f in parent_img_dir.glob("*")]
    output_dir = Path(output_dir_name)
    
    #create correctly labelled sub dirs
    for cls in classes:
        path = output_dir/'correct'/cls
        path.mkdir(parents = True, exist_ok = True)
        
    #create incorrectly labelled sub dirs
    for cls_correct in classes:
        path = output_dir/'incorrect'/f'lbl_{cls_correct}_pred_under-thres'
        path.mkdir(parents=True, exist_ok=True)
        for cls_wrong in classes:
            if cls_correct != cls_wrong:
                path = output_dir/'incorrect'/f'lbl_{cls_correct}_pred_{cls_wrong}'
                path.mkdir(parents=True, exist_ok=True)
    
def plot_confusion_matrix(cf_matrix, labels, plot_name):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.2g')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels, size=8)
    ax.yaxis.set_ticklabels(labels, size=8)
    plt.savefig(f"yolov5_evaluation/{plot_name}.png")

def cluster_invalid(labels):
    clustered_invalid_list = []
    for lbl in labels:
        if 'invalid' in lbl:
            new_lbl = 'invalid'
            clustered_invalid_list.append(new_lbl)
        else:
            clustered_invalid_list.append(lbl)
    return clustered_invalid_list

def evaluate_results(y_label, y_pred, average_inference_time, group_invalid):
    cf_labels = names
    report_name = 'report'
    cf_plot_name = 'cm'
    if group_invalid:
        y_label = cluster_invalid(y_label)
        y_pred = cluster_invalid(y_pred)
        cf_labels = ['positive', 'negative', 'invalid']
        report_name = "report_group_invalid"
        cf_plot_name = 'cm_group_invalid'
    
    #confusion matrix
    cf_matrix = confusion_matrix(y_label, y_pred, labels=cf_labels, normalize='true')
    plot_confusion_matrix(cf_matrix, cf_labels, plot_name=cf_plot_name)
    report = classification_report(y_label, y_pred)
    with open(f"yolov5_evaluation/{report_name}.txt", "w") as f:
        f.write(report)
        f.write(f"average inference time: {average_inference_time:.2f} ms")

if __name__ == "__main__":
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    
    y_pred = []
    y_label = []
    y_not_found = []
    time_takens = []
    
    parent_img_dir = Path(test_folder_path)
    create_evaluation_folder(parent_img_dir, output_dir_name=output_folder)
    imgs = [str(f) for f in parent_img_dir.glob("*/*.jpg")] + [str(f)  for f in parent_img_dir.glob("*/*.jpeg")] + [str(f) for f in parent_img_dir.glob("*/*.png")]
    
    for img in imgs:
        start_time = time.time()
        predicted_result, labelled_img = inference(session, img)
        end_time = time.time()
        
        time_taken = (end_time - start_time)*1000
        time_takens.append(time_taken)
        i_name = os.path.split(img)[-1].split('.')[0]
        correct_class_idx =  get_label_from_folder(img)
        
        imgname = img.split("\\")[-1]
        
        if predicted_result["class"] != None:
            if correct_class_idx == predicted_result["class"]:
                path = Path(output_folder)/'correct'/names[correct_class_idx]/imgname
                cv2.imwrite(str(path), labelled_img)
                print(f"Correct prediction {img} in {time_taken} ms")
            else:
                path = Path(output_folder)/'incorrect'/f'lbl_{names[correct_class_idx]}_pred_{names[predicted_result["class"]]}'/imgname
                cv2.imwrite(str(path), labelled_img)
                print(f"Incorrect prediction {img} in {time_taken} ms img saved to {str(path)}")

            y_pred.append(names[predicted_result["class"]])
            y_label.append(names[correct_class_idx])
        else:
            path = Path(output_folder)/'incorrect'/f'lbl_{names[correct_class_idx]}_pred_under-thres'/imgname
            cv2.imwrite(str(path), labelled_img)
            # y_pred.append('invalid-under-thres')
            # y_label.append(names[correct_class_idx])

        
    average_inference_time  = np.mean(time_takens)
        
    evaluate_results(y_label = y_label, 
                     y_pred  = y_pred, 
                     average_inference_time = average_inference_time, 
                     group_invalid = False)

        
        
