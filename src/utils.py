import yaml
import cv2
import os

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

def draw_results(frame, results, fps, config):
    """
    Vẽ bounding box và thông tin lên frame.
    
    Args:
        frame (numpy.ndarray): Frame hình ảnh.
        results (list): Kết quả từ model YOLO.
        fps (float): Giá trị FPS hiện tại.
        config (dict): Cấu hình hiển thị.
        
    Returns:
        numpy.ndarray: Frame đã được vẽ thông tin.
    """
    bbox_color = tuple(config['display']['bbox_color'])
    text_color = tuple(config['display']['text_color'])
    thickness = config['display']['box_thickness']
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, thickness)
            
            # Vẽ nhãn
            label = f"{confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

    # Hiển thị FPS (nếu cần vẽ trực tiếp lên frame, nhưng yêu cầu là hiển thị trên UI)
    # Tuy nhiên, vẽ lên frame cũng tốt cho debug hoặc recording
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame
