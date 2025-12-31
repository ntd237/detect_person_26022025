from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
from src.utils import draw_results

class ThreadStream(QThread):
    """
    Luồng hiển thị frame.
    Nhận frame và kết quả, vẽ bbox, chuyển đổi sang QImage và gửi tín hiệu cập nhật UI.
    """
    # Tín hiệu gửi QImage để hiển thị và FPS
    update_image = pyqtSignal(QImage, float)

    def __init__(self, config):
        """
        Khởi tạo luồng stream.
        
        Args:
            config (dict): Cấu hình hệ thống.
        """
        super().__init__()
        self.config = config
        self.running = True
        self.polygon_points = None  # Danh sách điểm polygon (nếu có)

    def set_polygon(self, polygon_points):
        """
        Thiết lập polygon để filter detection.
        
        Args:
            polygon_points (list): Danh sách các điểm polygon [(x, y), ...].
        """
        self.polygon_points = polygon_points

    def display_frame(self, frame, results, fps):
        """
        Slot nhận frame và kết quả từ process thread.
        """
        if not self.running:
            return

        # Vẽ bounding box và thông tin (có polygon filter nếu được set)
        frame = draw_results(frame, results, fps, self.config, self.polygon_points)

        # Chuyển đổi BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        
        # Tạo QImage
        qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Gửi tín hiệu cập nhật UI
        # Lưu ý: QImage chia sẻ bộ nhớ với numpy array, nên nếu frame thay đổi trước khi vẽ xong có thể lỗi.
        # Tuy nhiên trong luồng này ta tạo mới mỗi lần gọi hàm này.
        # Để an toàn hơn có thể dùng .copy()
        self.update_image.emit(qimg.copy(), fps)

    def stop(self):
        """Dừng luồng."""
        self.running = False
        self.quit()
        self.wait()
