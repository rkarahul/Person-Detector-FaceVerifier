import os
import torch
import numpy as np
import cv2
from flask import Flask, jsonify,request, send_file
import json, os, signal
import base64
from PIL import Image
import io
import time
from datetime import datetime
import torchvision
from collections import Counter
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import multiprocessing
import signal
import logging
import traceback
logger=logging.getLogger('my-logger')#--- Logging error ---
logger.propagate=False

class DefectData:
    def __init__(self, defType, coordinates):
        self.defType = defType
        self.coordinates = coordinates


class ObjectDetection:
    def __init__(self, weights):
        self.model = self.load_model(weights)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        #print("\n\nDevice Used:", self.device)

    def load_model(self, weights):
        model = torch.hub.load('yolov5', 'custom', path=weights,source='local',force_reload=True)
        #print("loaded_model")
        return model

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def box_iou(self,box1, box2, eps=1e-7):
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

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
        prediction,
        conf_thres=0.5,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0
        ):

        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

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
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                #print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    
    def prepro_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()  # convert to tensor
        img /= 255.0 
        # Run inference
        img = img.unsqueeze(0)  # add batch dimension
        img = img.to(self.device)
        return img
    

    def without_granding(self,img, thr=105):
        img = img[120:500,100:500]
        dst_avg = np.mean(img)
        #print("dst_avg :",dst_avg)
        if dst_avg < thr:
            return False#True
        return False

    def score_frame(self, frame,cam_code): 
        self.model.to(self.device)
        pro_img = self.prepro_image(frame)
        results = self.model(pro_img)       
        det = self.non_max_suppression(results, conf_thres=0.55, iou_thres=0.45)[0]#45
        label_color = {
            "SurfaceDefect": (255, 0, 0),
            "Root Grinding": (0, 0, 255),
            "ChamferMiss": (0, 0, 255),
            "Step Grinding": (0,0,255),
            "Flank Unclean": (0,0,255),
            "Handling Dent": (0,0,255),
            }
        class_ids=[]
        cords=[]
        lab_list=[] #labels
        cham=0

        for *xyxy, conf, cls in det:
            lab=self.model.names[int(cls)]
            cf=float(conf.item())
            #print("conf",cf)
            #print("lab",lab)
            center_y=int(xyxy[3])-int(xyxy[1])
            if int(xyxy[0])<=40 or int(xyxy[0])>=500:#handliong cornes dent(x cordinate)
                continue
            elif (lab =='Flank_unclean' or lab == "Rust") and int(xyxy[1])>=500 and center_y<50: #y-coordinate  # or Handling dent
                continue
            elif (lab =='Flank_unclean' or lab == "Rust") and  int(xyxy[1])<=140 and center_y<50:#lab =='Flank_unclean or Rust'
                continue
            elif lab =='Flank_unclean' and cf<0.58:#int(xyxy[1])<=140: #
                continue
            elif cam_code!=1 and int(xyxy[1])>=500:#update 11/4/024 for heat treatmennnt
                continue
            elif lab=="ChamferMiss" and int(xyxy[1])<=320:
                cham+=1
                label="Chamfer miss upper"
                if cham>2:
                    lab_list.append(label)
            elif lab=="ChamferMiss" and int(xyxy[1])>=320:
                cham+=1
                label="Chamfer miss lower"
                if cham>2:
                    lab_list.append(label)
            else: # All defect
                label = f'{lab} {conf:.2f}'
                lab_list.append(lab)
                 
            #bgr = label_color.get(lab, (0, 0, 0))
            class_ids.append(int(cls))
            cor=[lab,[int(xyxy[0]),int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]]
            #print("cords for single obj :",cor)
            cords.append(cor)
            #if label=="Chamfer miss upper" or label=="Chamfer miss lower":
            #    pass
            #else:
            if lab=="ChamferMiss" and cham<=2:
                pass
            else:
                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                if int(xyxy[1])>25:
                    frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[3]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
        #print("all cords : ",cords)
        #lab_list=list(set(lab_list))
        if len(lab_list)!=0:
            lab_list.insert(0,"Ng")
        else:
            lab_list.insert(0,"Ok")

        return lab_list,frame

def append_to_logger(text):
    with open("error.txt",'a') as logfile:
        logfile.write(text + "\n")


def pers_transf_1(img):
    pts1 = np.float32([[0, 460], [1180, 0], [0, 1830], [1180, 1435]])#cam1 lhs
    #cv2.imwrite("pers_transf_1.jpg",img)#top_left(x,y),top_right,bottom_left,bottom_right
    pts2 = np.float32([[0,0],[950,0],[0,610],[950,610]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(950,610))
    #dst = dst[:,100:850]
    return dst

def pers_transf_2(img):
    pts1 = np.float32([[0, 340], [1180, 0], [0, 1765], [1180, 1310]])#cam2 lhs
    #cv2.imwrite("pers_transf_2.jpg",img)#top_left(x,y),top_right,bottom_left,bottom_right
    pts2 = np.float32([[0,0],[950,0],[0,610],[950,610]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(950,610))
    #dst = dst[:,100:850]#:850
    return dst

# Load the model

detection = ObjectDetection(weights=r'./yolov5/best.pt')
# temp detection
imgdtt=cv2.imread("temp.bmp")
imgdtt = cv2.resize(imgdtt, (640, 640))
defls, det_imgdtt = detection.score_frame(imgdtt,1)

def serialize_defect(defect):
    return {
        "defType": defect.defType,
        "coordinates": defect.coordinates
    }

app1 = FastAPI()

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def opencv_image_to_base64(image):
    # Encode the OpenCV image as a JPEG image in memory
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert the image buffer to a base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string

##def shutdown_server(self):
##    os.kill(os.getpid(), signal.SIGINT)
##    return jsonify({"success": True, "message": "Server is shutting down..."})
##    
##@app1.route('/shutdown')
##def shutdown():
##    shutdown_server()
##    return 'Server shutting down...'

def shutdown_server():
    #print("Shutting down server...")
    os.kill(os.getpid(), signal.SIGINT)
    
server_process_id=os.getpid()
@app1.get('/shutdown')   
def shutdown():
    try:
        os.kill(server_process_id,signal.SIGINT)
        message="Server shutting down..."
    except ProcessLookupError:
        message="Server process not found..."
    return JSONResponse(content={"message": message})#, status_code=200)

@app1.get("/")
def home():
    return {"Health": "OK"}

@app1.get("/ServerCheck")
def ServerCheck():
    #print("In ServerCheck")
    return {"Server": "OK"}

@app1.post('/predict_cover')
def predictions(data: dict):
    try:
        #print("request recived")
        #append_to_logger("request recived")
        #data = json.loads(file)
        image_bytes=data.get("image")
        thr=data.get("thr")
        cam_code=data.get("camCode")
        #print("cam_code :",cam_code)
        imgFromcs = stringToImage(image_bytes)
        numpyImage = np.array(imgFromcs)
        #print("Image converted to numpy successfully on server")

        start = time.time()
        openCvImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2BGR)
        if cam_code==1:
            frame = pers_transf_1(openCvImage)  # correction perspective cam 1
        else:
            frame = pers_transf_2(openCvImage)  # correction perspective cam 2
        frame = cv2.resize(frame, (640, 640))
        time_s=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        f_path=r"D:\Software\HelicalGearInspection\HelicalGearInspection\Images"
        file_nam=os.path.join(f_path,f"image_{time_s}.jpg")
        #cv2.imwrite(file_nam,frame)
        # Rotate the image by 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        res = detection.without_granding(frame,thr)
        if res:
            frame = cv2.rectangle(frame,(100, 100), (500, 500), (255, 0, 0), 2)
            frame = cv2.putText(frame, 'Without_Grinding', (200,200 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
            defList, det_frame = ['Ng', 'Without_Grinding'], frame
        else:
            defList, det_frame = detection.score_frame(frame,cam_code)
        #print("defList : ", defList) 
        base64Image = opencv_image_to_base64(det_frame)
        #print("detections with label :", defList)
        return {"defImage": base64Image, "serialized_Defects": defList}  # JSON response
    except Exception as ex:
        #print("Exception in predict_cover ", ex)
        base64Image = opencv_image_to_base64(frame)#temp img
        #print("Ex: Image Converted to B64")
        expt=traceback.format_exc()
        append_to_logger(expt)
        return {"defImage": base64Image, "serialized_Defects": ["OK"]}  # JSON
        #raise HTTPException(status_code=500, detail=str(ex))

if __name__ == '__main__':
    uvicorn.run(app1, host="127.0.0.1", port=5000, workers=4) #multiprocessing.cpu_count() #, workers=4

"""
#data = json.loads(request.data)
app1 = Flask(__name__)
def serialize_defect(defect):
    return {
        "defType": defect.defType,
        "coordinates": defect.coordinates
    }

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def opencv_image_to_base64(image):
    # Encode the OpenCV image as a JPEG image in memory
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert the image buffer to a base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string

# Define Flask routes

def shutdown_server():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"success": True, "message": "Server is shutting down..."})
    
@app1.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app1.route("/ServerCheck")
def ServerCheck():
    print("In ServerCheck")
    return {"Server": "OK"}

@app1.route("/")
def home():
    return {"Health": "OK"}

@app1.route('/predict_cover', methods=['POST'])
def predictions():
    try:
        print("In predict_cover")
        file = json.loads(request.data)# request.files['image']
        #print(file)
        image_bytes = file
        imgFromcs = stringToImage(image_bytes)
        numpyImage = np.array(imgFromcs)
        print("Image converted to numpy successfuly orinf server")

        start=time.time()
        openCvImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2BGR)
        frame=pers_transf(openCvImage) #correction prespective 
        frame = cv2.resize(frame, (640,640))
        res = detection.without_granding(frame)
        if res:
            defList,det_frame=['Ng','without_granding'],frame
        else:
            defList,det_frame = detection.score_frame(frame)
        print("defList : ",defList)
        base64Image = opencv_image_to_base64(det_frame)
        print("detections with lablel :",defList)
        return jsonify(defImage=base64Image, serialized_Defects=defList)
    except Exception as ex:
        print("Exception in predict_cover ", ex)
        return ex

if __name__ == '__main__':
    #app.run()
    app1.run(port=5000)


def pers_transf(img):
  
  #pts1 = np.float32([[0,480],[1180,0],[0,1800],[1200,1550]])
  pts1 = np.float32([[0, 445], [1175, 135], [0, 1680], [1150, 1280]])
  #top_left(x,y),top_right,bottom_left,bottom_right
  pts2 = np.float32([[0,0],[950,0],[0,610],[950,610]])
  M = cv2.getPerspectiveTransform(pts1,pts2)
  dst = cv2.warpPerspective(img,M,(950,610))
  dst = dst[:,100:850]
  return dst"""
