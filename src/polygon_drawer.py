"""
Polygon Drawer - UI vẽ polygon trên frame đầu tiên của video.
Người dùng sử dụng chuột để vẽ các điểm polygon.
"""
import cv2
import sys
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QPolygon, QColor, QBrush
import numpy as np


class PolygonDrawer(QDialog):
    """
    Dialog PyQt5 để vẽ polygon trên frame đầu tiên của video.
    
    Cách sử dụng:
        - Click trái: Thêm điểm mới
        - Click phải hoặc Enter: Hoàn thành vẽ polygon
        - Phím C: Clear tất cả điểm và vẽ lại
        - Phím Esc: Hủy và thoát
    """
    
    def __init__(self, config, parent=None):
        """
        Khởi tạo Polygon Drawer.
        
        Args:
            config (dict): Cấu hình hệ thống, cần có 'video.path' và 'polygon' settings.
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        self.config = config
        self.polygon_points = []  # Danh sách các điểm polygon [(x, y), ...]
        self.original_frame = None  # Frame gốc từ video
        self.display_frame = None  # Frame hiển thị với polygon
        
        # Lấy cấu hình polygon
        polygon_config = config.get('polygon', {})
        self.line_color = tuple(polygon_config.get('line_color', [0, 255, 255]))
        self.line_thickness = polygon_config.get('line_thickness', 2)
        self.fill_alpha = polygon_config.get('fill_alpha', 0.2)
        
        # Load frame đầu tiên
        self._load_first_frame()
        
        if self.original_frame is None:
            raise RuntimeError("Không thể load frame đầu tiên từ video")
        
        # Khởi tạo UI
        self._init_ui()
    
    def _load_first_frame(self):
        """Load frame đầu tiên từ video."""
        video_path = self.config['video']['path']
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.original_frame = frame.copy()
                self.display_frame = frame.copy()
        cap.release()
    
    def _init_ui(self):
        """Khởi tạo giao diện."""
        self.setWindowTitle("Vẽ Polygon - Click trái để thêm điểm, Click phải/Enter để hoàn thành")
        self.setModal(True)
        
        # Lấy kích thước frame
        h, w = self.original_frame.shape[:2]
        
        # Giới hạn kích thước cửa sổ nếu frame quá lớn
        max_width = 1280
        max_height = 720
        scale = min(max_width / w, max_height / h, 1.0)
        self.display_width = int(w * scale)
        self.display_height = int(h * scale)
        self.scale_factor = scale
        
        # Image label
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(self.display_width, self.display_height)
        self.image_label.setMouseTracking(True)
        
        # Instruction label
        self.instruction_label = QLabel(
            "Click trái: Thêm điểm | Click phải/Enter: Hoàn thành | C: Clear | Esc: Hủy",
            self
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #333; color: white;")
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.instruction_label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Hiển thị frame ban đầu
        self._update_display()
    
    def _update_display(self):
        """Cập nhật hiển thị frame với polygon."""
        # Copy frame gốc
        frame = self.original_frame.copy()
        
        # Vẽ các điểm đã chọn
        points = self.polygon_points
        if len(points) > 0:
            # Vẽ các điểm
            for i, point in enumerate(points):
                cv2.circle(frame, point, 6, (0, 0, 255), -1)  # Điểm màu đỏ
                cv2.circle(frame, point, 6, (255, 255, 255), 2)  # Viền trắng
                
                # Vẽ số thứ tự
                cv2.putText(frame, str(i + 1), (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Vẽ các đường nối
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], self.line_color, self.line_thickness)
            
            # Nếu có >= 3 điểm, vẽ polygon fill và đường nối cuối
            if len(points) >= 3:
                # Vẽ đường nối điểm cuối về điểm đầu (dashed line để phân biệt)
                cv2.line(frame, points[-1], points[0], self.line_color, self.line_thickness)
                
                # Vẽ fill polygon với alpha
                overlay = frame.copy()
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], self.line_color)
                cv2.addWeighted(overlay, self.fill_alpha, frame, 1 - self.fill_alpha, 0, frame)
        
        # Resize frame nếu cần
        if self.scale_factor < 1.0:
            frame = cv2.resize(frame, (self.display_width, self.display_height))
        
        # Chuyển đổi sang QPixmap và hiển thị
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
    
    def mousePressEvent(self, event):
        """Xử lý sự kiện click chuột."""
        # Chỉ xử lý click trong vùng image_label
        pos = self.image_label.mapFromGlobal(event.globalPos())
        
        if pos.x() < 0 or pos.y() < 0 or pos.x() >= self.display_width or pos.y() >= self.display_height:
            return
        
        # Chuyển đổi tọa độ về frame gốc
        x = int(pos.x() / self.scale_factor)
        y = int(pos.y() / self.scale_factor)
        
        if event.button() == Qt.LeftButton:
            # Thêm điểm mới
            self.polygon_points.append((x, y))
            self._update_display()
            
        elif event.button() == Qt.RightButton:
            # Hoàn thành vẽ polygon
            self._finish_drawing()
    
    def keyPressEvent(self, event):
        """Xử lý sự kiện phím."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Hoàn thành vẽ polygon
            self._finish_drawing()
            
        elif event.key() == Qt.Key_C:
            # Clear tất cả điểm
            self.polygon_points = []
            self._update_display()
            
        elif event.key() == Qt.Key_Escape:
            # Hủy và thoát
            self.reject()
    
    def _finish_drawing(self):
        """Hoàn thành vẽ polygon và đóng dialog."""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(
                self,
                "Không đủ điểm",
                "Polygon cần ít nhất 3 điểm. Hiện tại có {} điểm.".format(len(self.polygon_points))
            )
            return
        
        self.accept()
    
    def get_polygon_points(self):
        """
        Lấy danh sách các điểm polygon.
        
        Returns:
            list: Danh sách các điểm [(x1, y1), (x2, y2), ...]
        """
        return self.polygon_points.copy()


def show_polygon_drawer(config):
    """
    Hiển thị dialog vẽ polygon và trả về danh sách điểm.
    
    Args:
        config (dict): Cấu hình hệ thống.
        
    Returns:
        list or None: Danh sách điểm polygon nếu thành công, None nếu hủy.
    """
    drawer = PolygonDrawer(config)
    result = drawer.exec_()
    
    if result == QDialog.Accepted:
        return drawer.get_polygon_points()
    return None
