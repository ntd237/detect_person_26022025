"""
SAHI - Slicing Aided Hyper Inference cho YOLO.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple


class SAHIDetectionResult:
    """
    Class chứa kết quả detection từ SAHI, tương thích với format của draw_results().
    """
    
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, class_id: int):
        """
        Khởi tạo kết quả detection.
        
        Args:
            bbox (tuple): (x1, y1, x2, y2) tọa độ bounding box.
            confidence (float): Độ tin cậy của detection.
            class_id (int): ID của class được detect.
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


class SAHIResultWrapper:
    """
    Wrapper để chứa danh sách kết quả SAHI, tương thích với format ultralytics.
    """
    
    def __init__(self, detections: List[SAHIDetectionResult]):
        """
        Args:
            detections (list): Danh sách SAHIDetectionResult.
        """
        self.detections = detections
        self.boxes = SAHIBoxesWrapper(detections)


class SAHIBoxesWrapper:
    """
    Wrapper cho boxes để tương thích với ultralytics format (result.boxes).
    """
    
    def __init__(self, detections: List[SAHIDetectionResult]):
        self.detections = detections
    
    def __iter__(self):
        """Cho phép iterate qua các box giống như ultralytics."""
        for det in self.detections:
            yield SAHIBoxWrapper(det)
    
    def __len__(self):
        return len(self.detections)


class SAHIBoxWrapper:
    """
    Wrapper cho single box để tương thích với ultralytics format.
    """
    
    def __init__(self, detection: SAHIDetectionResult):
        self._detection = detection
        # Tạo tensor-like object cho xyxy và conf
        self.xyxy = [detection.bbox]  # List chứa tuple (x1,y1,x2,y2)
        self.conf = [ConfWrapper(detection.confidence)]
    
    @property
    def cls(self):
        return [self._detection.class_id]


class ConfWrapper:
    """Wrapper cho confidence value để có method .item()."""
    
    def __init__(self, value: float):
        self._value = value
    
    def item(self):
        return self._value


class SAHI:
    """
    SAHI - Slicing Aided Hyper Inference.
    
    Chia ảnh thành nhiều slice nhỏ, chạy inference trên từng slice,
    sau đó merge kết quả bằng NMS để loại bỏ duplicate detections.
    """
    
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, 
                 imgsz: int = 640, device: str = "cuda", target_classes: List[int] = None,
                 use_ios: bool = True, postprocess_type: str = "nmm"):
        """
        Khởi tạo SAHI.
        
        Args:
            model_path (str): Đường dẫn đến file model YOLO.
            conf (float): Ngưỡng confidence cho detection.
            iou (float): Ngưỡng IOU/IOS cho NMS/NMM merge các slice.
            imgsz (int): Kích thước input của model.
            device (str): Device để chạy inference ('cuda' hoặc 'cpu').
            target_classes (list): Danh sách class ID cần detect (None = tất cả).
            use_ios (bool): Sử dụng IOS (Intersection Over Smallest) thay vì IoU.
                           IOS hoạt động tốt hơn khi box bị cắt ở biên slice.
            postprocess_type (str): Loại postprocess - "nms" hoặc "nmm".
                           NMS: loại bỏ box chồng lấn, giữ box confidence cao nhất.
                           NMM: merge các box chồng lấn thành 1 box lớn hơn.
        """
        self.model = YOLO(model_path)
        if model_path.endswith(".pt"):
            self.model.to(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.target_classes = target_classes
        self.use_ios = use_ios
        self.postprocess_type = postprocess_type.lower()
        
        metric = "IOS" if use_ios else "IoU"
        print(f"[SAHI] Khởi tạo thành công - conf={conf}, {metric}_thresh={iou}, imgsz={imgsz}, postprocess={self.postprocess_type.upper()}")
    
    def _get_slices(self, img_shape: Tuple[int, int], n_slices: int, 
                    overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
        """
        Tính toán các slice dựa trên kích thước ảnh.
        Chia theo chiều dài hơn của ảnh.
        
        Args:
            img_shape (tuple): Shape của ảnh (h, w, ...).
            n_slices (int): Số lượng slice.
            overlap (float): Tỷ lệ overlap giữa các slice.
            
        Returns:
            list: Danh sách các slice (x1, y1, x2, y2).
        """
        h, w = img_shape[:2]
        vertical = h > w  # Chia theo chiều dọc nếu ảnh portrait
        
        if vertical:
            # Chia theo chiều dọc (theo height)
            slice_h = h // n_slices
            overlap_h = int(slice_h * overlap)
            slices = []
            for i in range(n_slices):
                y1 = max(0, i * slice_h - (overlap_h if i > 0 else 0))
                y2 = min(h, (i + 1) * slice_h + overlap_h)
                slices.append((0, y1, w, y2))
        else:
            # Chia theo chiều ngang (theo width)
            slice_w = w // n_slices
            overlap_w = int(slice_w * overlap)
            slices = []
            for i in range(n_slices):
                x1 = max(0, i * slice_w - (overlap_w if i > 0 else 0))
                x2 = min(w, (i + 1) * slice_w + overlap_w)
                slices.append((x1, 0, x2, h))
        
        return slices
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Non-Maximum Suppression để loại bỏ duplicate detections.
        Hỗ trợ cả IoU và IOS metric.
        
        Args:
            boxes (np.ndarray): Mảng các box (N, 4).
            scores (np.ndarray): Mảng confidence scores (N,).
            threshold (float): Ngưỡng IoU/IOS.
            
        Returns:
            np.ndarray: Indices của các box được giữ lại.
        """
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
            
            if self.use_ios:
                # IOS = Intersection Over Smallest
                # Hiệu quả hơn khi box bị cắt ở biên slice (box nhỏ hơn sẽ bị merge)
                min_areas = np.minimum(areas[i], areas[order[1:]])
                overlap = inter / (min_areas + 1e-6)  # Tránh chia cho 0
            else:
                # IoU = Intersection Over Union (công thức gốc)
                overlap = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Sử dụng < để đảm bảo loại bỏ các box có overlap >= threshold
            order = order[np.where(overlap < threshold)[0] + 1]
        
        return np.array(keep)
    
    def _nmm(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Non-Maximum Merging - merge các box chồng lấn thành 1 box lớn hơn.
        Khác với NMS (loại bỏ), NMM sẽ gộp các box thành 1 box bao trọn.
        
        Args:
            boxes (np.ndarray): Mảng các box (N, 4) định dạng [x1, y1, x2, y2].
            scores (np.ndarray): Mảng confidence scores (N,).
            threshold (float): Ngưỡng overlap để quyết định merge.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (merged_boxes, merged_scores)
        """
        if len(boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([])
        
        # Copy để không modify array gốc
        boxes = boxes.copy()
        scores = scores.copy()
        
        merged_boxes = []
        merged_scores = []
        
        # Đánh dấu box đã được merge
        used = np.zeros(len(boxes), dtype=bool)
        
        # Sắp xếp theo confidence giảm dần
        order = scores.argsort()[::-1]
        
        for idx in order:
            if used[idx]:
                continue
            
            # Bắt đầu nhóm merge với box hiện tại
            current_box = boxes[idx].copy()
            current_scores = [scores[idx]]
            used[idx] = True
            
            # Tìm tất cả box chưa dùng có overlap với current_box
            for other_idx in order:
                if used[other_idx]:
                    continue
                
                other_box = boxes[other_idx]
                
                # Tính overlap (IOS hoặc IoU)
                xx1 = max(current_box[0], other_box[0])
                yy1 = max(current_box[1], other_box[1])
                xx2 = min(current_box[2], other_box[2])
                yy2 = min(current_box[3], other_box[3])
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                
                area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                area_other = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
                
                if self.use_ios:
                    # IOS = Intersection Over Smallest
                    min_area = min(area_current, area_other)
                    overlap = inter / (min_area + 1e-6)
                else:
                    # IoU
                    overlap = inter / (area_current + area_other - inter + 1e-6)
                
                if overlap >= threshold:
                    # Merge: mở rộng current_box để bao trọn other_box
                    current_box[0] = min(current_box[0], other_box[0])
                    current_box[1] = min(current_box[1], other_box[1])
                    current_box[2] = max(current_box[2], other_box[2])
                    current_box[3] = max(current_box[3], other_box[3])
                    current_scores.append(scores[other_idx])
                    used[other_idx] = True
            
            merged_boxes.append(current_box)
            # Confidence của box merged = trung bình các confidence
            merged_scores.append(np.mean(current_scores))
        
        if not merged_boxes:
            return np.array([]).reshape(0, 4), np.array([])
        
        return np.array(merged_boxes), np.array(merged_scores)
    def predict_raw(self, img: np.ndarray, n_slices: int = 2, 
                    overlap: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Thực hiện sliced prediction và trả về raw arrays.
        
        Args:
            img (np.ndarray): Frame hình ảnh BGR.
            n_slices (int): Số lượng slice.
            overlap (float): Tỷ lệ overlap.
            
        Returns:
            tuple: (boxes, scores, classes) dạng numpy arrays.
        """
        h, w = img.shape[:2]
        slices = self._get_slices(img.shape, n_slices, overlap)
        
        all_boxes, all_scores, all_classes = [], [], []
        
        for x1, y1, x2, y2 in slices:
            crop = img[y1:y2, x1:x2]
            
            # Inference trên crop với target classes filter
            results = self.model.predict(
                crop, 
                conf=self.conf, 
                imgsz=self.imgsz,
                classes=self.target_classes,
                verbose=False
            )[0]
            
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                # Chuyển đổi tọa độ box về ảnh gốc
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
        
        # Áp dụng NMS hoặc NMM theo từng class dựa vào config
        unique_classes = np.unique(all_classes)
        final_boxes, final_scores, final_classes = [], [], []
        
        for cls in unique_classes:
            mask = all_classes == cls
            cls_boxes = all_boxes[mask]
            cls_scores = all_scores[mask]
            
            if self.postprocess_type == "nmm":
                # Non-Maximum Merging: merge các box chồng lấn
                merged_boxes, merged_scores = self._nmm(cls_boxes, cls_scores, self.iou)
                if len(merged_boxes) > 0:
                    final_boxes.append(merged_boxes)
                    final_scores.append(merged_scores)
                    final_classes.append(np.full(len(merged_boxes), cls))
            else:
                # Non-Maximum Suppression: loại bỏ box chồng lấn
                keep = self._nms(cls_boxes, cls_scores, self.iou)
                if len(keep) > 0:
                    final_boxes.append(cls_boxes[keep])
                    final_scores.append(cls_scores[keep])
                    final_classes.append(np.full(len(keep), cls))
        
        if not final_boxes:
            return np.array([]), np.array([]), np.array([])
        
        return np.vstack(final_boxes), np.concatenate(final_scores), np.concatenate(final_classes)
    
    def predict(self, img: np.ndarray, n_slices: int = 2, overlap: float = 0.2) -> List:
        """
        Thực hiện sliced prediction và trả về kết quả tương thích với ultralytics format.
        
        Args:
            img (np.ndarray): Frame hình ảnh BGR.
            n_slices (int): Số lượng slice.
            overlap (float): Tỷ lệ overlap.
            
        # Returns:
            list: Kết quả detection theo format ultralytics.
        """
        boxes, scores, classes = self.predict_raw(img, n_slices, overlap)
        
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            detections.append(SAHIDetectionResult(
                bbox=tuple(box),
                confidence=float(score),
                class_id=int(cls)
            ))
        
        return [SAHIResultWrapper(detections)]


class SAHIWrapper:
    """
    Wrapper class để tích hợp SAHI vào ThreadProcess với config từ YAML.
    """
    
    def __init__(self, model_path: str, device: str, conf_threshold: float, 
                 target_classes: list, imgsz: int, sahi_config: dict):
        """
        Khởi tạo SAHI wrapper từ config.
        
        Args:
            model_path (str): Đường dẫn đến file model YOLO.
            device (str): Device để chạy inference.
            conf_threshold (float): Ngưỡng confidence cho detection.
            target_classes (list): Danh sách class ID cần detect.
            imgsz (int): Kích thước input của model.
            sahi_config (dict): Cấu hình SAHI từ config.yaml.
        """
        # Lấy các tham số SAHI từ config
        self.n_slices = sahi_config.get('n_slices', 3)
        self.overlap = sahi_config.get('overlap', 0.2)
        iou_threshold = sahi_config.get('iou_threshold', 0.45)
        use_ios = sahi_config.get('use_ios', True)
        postprocess_type = sahi_config.get('postprocess_type', 'nmm')  # nmm hoặc nms
        
        # Khởi tạo SAHI core
        self.sahi = SAHI(
            model_path=model_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            target_classes=target_classes,
            use_ios=use_ios,
            postprocess_type=postprocess_type
        )
        
        metric = "IOS" if use_ios else "IoU"
        print(f"[SAHIWrapper] Khởi tạo thành công với n_slices={self.n_slices}, overlap={self.overlap}, metric={metric}, postprocess={postprocess_type.upper()}")
    
    def predict(self, frame: np.ndarray) -> List:
        """
        Thực hiện sliced prediction trên frame.
        
        Args:
            frame (np.ndarray): Frame hình ảnh BGR.
            
        Returns:
            list: Kết quả detection theo format ultralytics.
        """
        return self.sahi.predict(frame, n_slices=self.n_slices, overlap=self.overlap)