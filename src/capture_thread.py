import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal

class ThreadCapture(QThread):
    """
    Luồng đọc frame từ video file.
    """
    # Tín hiệu gửi frame mới (numpy array)
    new_frame = pyqtSignal(object)

    def __init__(self, config):
        """
        Khởi tạo luồng capture.
        
        Args:
            config (dict): Cấu hình hệ thống.
        """
        super().__init__()
        self.video_path = config['video']['path']
        self.target_fps = config['video']['target_fps']
        self.cap = None
        self.running = True

    def run(self):
        """
        Hàm chạy chính của luồng.
        Đọc video và emit frame theo FPS mục tiêu.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"Không thể mở video: {self.video_path}")
            self.running = False
            return

        while self.running and self.cap.isOpened():
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # Nếu hết video, có thể loop lại hoặc dừng
                # Ở đây ta dừng
                break

            self.new_frame.emit(frame)
            
            elapsed_time = time.time() - start_time
            # Tính toán thời gian sleep để duy trì FPS ổn định
            sleep_time = max(0, (1.0 / self.target_fps) - elapsed_time)
            time.sleep(sleep_time)
            
        self.cap.release()

    def stop(self):
        """Dừng luồng an toàn."""
        self.running = False
        self.quit()
        self.wait()
