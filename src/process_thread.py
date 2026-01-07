import time
import torch
import queue
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from src.utils import draw_results

class ThreadProcess(QThread):
    """
    Luồng xử lý AI (Inference).
    """
    # Tín hiệu gửi frame gốc, kết quả detection và FPS xử lý
    processed_results = pyqtSignal(object, object, float)

    def __init__(self, config):
        """
        Khởi tạo luồng xử lý.
        
        Args:
            config (dict): Cấu hình hệ thống.
        """
        super().__init__()
        self.config = config
        self.model_path = config['model']['path']
        self.conf_threshold = config['model']['confidence_threshold']
        self.target_classes = config['model']['classes']
        self.input_size = config['model'].get('input_size', 640)  # Lấy input_size từ config, mặc định 640
        self.device = config['model']['device'] if torch.cuda.is_available() else "cpu"
        
        # Khởi tạo model
        self.model = YOLO(self.model_path)
        if self.model_path.endswith(".pt"):
            self.model.to(self.device)
        
        self.running = True
        self.frame_queue = queue.Queue(maxsize=5)  # Giới hạn queue để tránh lag
        self.prev_time = 0  # Thời gian xử lý frame trước đó

    def add_frame(self, frame):
        """
        Thêm frame vào hàng đợi để xử lý.
        Nếu hàng đợi đầy, frame cũ sẽ bị drop (hoặc frame mới không được thêm vào tùy chiến lược).
        Ở đây ta chọn drop frame mới nếu queue đầy để đảm bảo realtime.
        """
        if not self.running:
            return
        
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def run(self):
        """
        Vòng lặp chính xử lý frame.
        """
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Tính FPS throughput (số frame xử lý được trên giây thực tế)
                curr_time = time.time()
                if self.prev_time == 0:
                    fps = 0.0
                else:
                    # Tránh chia cho 0 nếu xử lý quá nhanh
                    delta = curr_time - self.prev_time
                    fps = 1.0 / delta if delta > 0 else 0.0
                self.prev_time = curr_time
                
                # Inference
                with torch.inference_mode():
                    results = self.model(frame, device=self.device, classes=self.target_classes, conf=self.conf_threshold, imgsz=self.input_size, verbose=False)
                
                # Gửi kết quả (frame gốc + results)
                self.processed_results.emit(frame, results, fps)
            else:
                # Sleep ngắn để tránh chiếm dụng CPU khi không có frame
                time.sleep(0.001)

    def stop(self):
        """Dừng luồng."""
        self.running = False
        self.quit()
        self.wait()
