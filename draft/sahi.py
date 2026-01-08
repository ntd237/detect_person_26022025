import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class SAHI:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
    
    def _get_slices(self, img_shape: Tuple[int, int], n_slices: int, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
        h, w = img_shape[:2]
        vertical = h > w
        
        if vertical:
            slice_h = h // n_slices
            overlap_h = int(slice_h * overlap)
            slices = []
            for i in range(n_slices):
                y1 = max(0, i * slice_h - (overlap_h if i > 0 else 0))
                y2 = min(h, (i + 1) * slice_h + overlap_h)
                slices.append((0, y1, w, y2))
        else:
            slice_w = w // n_slices
            overlap_w = int(slice_w * overlap)
            slices = []
            for i in range(n_slices):
                x1 = max(0, i * slice_w - (overlap_w if i > 0 else 0))
                x2 = min(w, (i + 1) * slice_w + overlap_w)
                slices.append((x1, 0, x2, h))
        
        return slices
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            order = order[np.where(iou <= iou_thresh)[0] + 1]
        
        return np.array(keep)
    
    def predict(self, img: np.ndarray, n_slices: int = 2, overlap: float = 0.2):
        h, w = img.shape[:2]
        slices = self._get_slices(img.shape, n_slices, overlap)
        
        all_boxes, all_scores, all_classes = [], [], []
        
        for x1, y1, x2, y2 in slices:
            crop = img[y1:y2, x1:x2]
            results = self.model.predict(crop, conf=self.conf, verbose=False)[0]
            
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)
                all_scores.append(results.boxes.conf.cpu().numpy())
                all_classes.append(results.boxes.cls.cpu().numpy())
        
        if not all_boxes:
            return np.array([]), np.array([]), np.array([])
        
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        
        unique_classes = np.unique(all_classes)
        final_boxes, final_scores, final_classes = [], [], []
        
        for cls in unique_classes:
            mask = all_classes == cls
            cls_boxes = all_boxes[mask]
            cls_scores = all_scores[mask]
            
            keep = self._nms(cls_boxes, cls_scores, self.iou)
            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_classes.append(np.full(len(keep), cls))
        
        return np.vstack(final_boxes), np.concatenate(final_scores), np.concatenate(final_classes)