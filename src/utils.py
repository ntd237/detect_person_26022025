import yaml
import cv2
import os
import numpy as np


def load_config(config_path="resources/configs/config.yaml"):
    """
    Tải cấu hình từ file YAML.
    
    Args:
        config_path (str): Đường dẫn đến file config.
        
    Returns:
        dict: Dictionary chứa cấu hình.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def is_bbox_in_polygon(bbox, polygon_points):
    """
    Kiểm tra xem center của bbox có nằm trong polygon không.
    
    Args:
        bbox (tuple): (x1, y1, x2, y2) tọa độ bounding box.
        polygon_points (list): List các điểm polygon [(x,y), ...].
        
    Returns:
        bool: True nếu center của bbox nằm trong polygon.
    """
    if not polygon_points or len(polygon_points) < 3:
        return True  # Không có polygon thì luôn return True
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Chuyển polygon points thành numpy array
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Sử dụng cv2.pointPolygonTest để kiểm tra
    # Return > 0: inside, = 0: on edge, < 0: outside
    result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
    
    return result >= 0


def draw_polygon(frame, polygon_points, config):
    """
    Vẽ polygon lên frame.
    
    Args:
        frame (numpy.ndarray): Frame hình ảnh.
        polygon_points (list): List các điểm polygon [(x,y), ...].
        config (dict): Cấu hình hiển thị.
        
    Returns:
        numpy.ndarray: Frame đã vẽ polygon.
    """
    if not polygon_points or len(polygon_points) < 3:
        return frame
    
    polygon_config = config.get('polygon', {})
    line_color = tuple(polygon_config.get('line_color', [0, 255, 255]))
    line_thickness = polygon_config.get('line_thickness', 2)
    fill_alpha = polygon_config.get('fill_alpha', 0.2)
    
    # Chuyển polygon points thành numpy array
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Vẽ fill polygon với alpha
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], line_color)
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    
    # Vẽ đường viền polygon
    cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=line_thickness)
    
    return frame


def draw_results(frame, results, fps, config, polygon_points=None):
    """
    Vẽ bounding box và thông tin lên frame.
    
    Args:
        frame (numpy.ndarray): Frame hình ảnh.
        results (list): Kết quả từ model YOLO.
        fps (float): Giá trị FPS hiện tại.
        config (dict): Cấu hình hiển thị.
        polygon_points (list, optional): List các điểm polygon để filter detection.
        
    Returns:
        numpy.ndarray: Frame đã được vẽ thông tin.
    """
    bbox_color = tuple(config['display']['bbox_color'])
    text_color = tuple(config['display']['text_color'])
    thickness = config['display']['box_thickness']
    
    # Vẽ polygon lên frame nếu có
    if polygon_points and len(polygon_points) >= 3:
        frame = draw_polygon(frame, polygon_points, config)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            
            # Kiểm tra xem bbox có trong polygon không (nếu có polygon)
            if polygon_points and len(polygon_points) >= 3:
                if not is_bbox_in_polygon((x1, y1, x2, y2), polygon_points):
                    continue  # Bỏ qua detection ngoài polygon
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, thickness)
            
            # Vẽ nhãn
            label = f"{confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

    # Hiển thị FPS (nếu cần vẽ trực tiếp lên frame)
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame
