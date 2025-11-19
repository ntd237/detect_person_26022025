from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
import time

class MainWindow(QWidget):
    """
    Giao diện chính của ứng dụng.
    """
    def __init__(self, controller, config):
        super().__init__()
        self.controller = controller
        self.config = config
        self.init_ui()
        
        # Kết nối tín hiệu từ Stream thread để cập nhật UI
        self.controller.stream_thread.update_image.connect(self.update_view)
        
        # Timer để cập nhật FPS hiển thị (nếu muốn cập nhật chậm hơn realtime)
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps_label)
        self.fps_timer.start(int(config['app']['fps_update_interval'] * 1000))
        
        self.frame_count = 0
        self.last_fps_time = time.time()

    def init_ui(self):
        """Khởi tạo giao diện."""
        self.setWindowTitle(self.config['app']['name'])
        self.resize(self.config['app']['width'], self.config['app']['height'])

        # Video View
        self.video_view = QGraphicsView(self)
        self.scene = QGraphicsScene()
        self.video_view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # FPS Label
        self.fps_label = QLabel("FPS: 0.00", self)
        self.fps_label.setStyleSheet("font-size: 16px; color: red; font-weight: bold;")
        self.fps_label.setAlignment(Qt.AlignRight)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_view)
        layout.addWidget(self.fps_label)
        self.setLayout(layout)

    def update_view(self, qimg, fps):
        """
        Cập nhật hình ảnh trên giao diện.
        
        Args:
            qimg (QImage): Hình ảnh đã xử lý.
            fps (float): FPS hiện tại (từ process thread).
        """
        self.frame_count += 1
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale pixmap theo kích thước cửa sổ (nếu cần)
        # Ở đây ta dùng fitInView của QGraphicsView để tự động scale
        self.pixmap_item.setPixmap(pixmap)
        self.video_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def update_fps_label(self):
        """Cập nhật nhãn FPS mỗi giây dựa trên số frame thực tế hiển thị."""
        curr_time = time.time()
        elapsed = curr_time - self.last_fps_time
        
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.2f}")
            
        self.frame_count = 0
        self.last_fps_time = curr_time

    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ."""
        self.controller.stop()
        event.accept()
