import sys
from PyQt5.QtWidgets import QApplication, QDialog
from src.controller import Controller
from src.gui import MainWindow
from src.utils import load_config


def main():
    """
    Điểm vào chính của ứng dụng.
    """
    # Tải cấu hình
    try:
        config = load_config("resources/configs/config.yaml")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        return

    app = QApplication(sys.argv)
    
    polygon_points = None
    
    # Nếu polygon mode được bật, yêu cầu vẽ polygon trước khi chạy app
    if config.get('polygon', {}).get('enabled', False):
        try:
            from src.polygon_drawer import PolygonDrawer
            drawer = PolygonDrawer(config)
            result = drawer.exec_()
            
            if result == QDialog.Accepted:
                polygon_points = drawer.get_polygon_points()
                print(f"Polygon đã được vẽ với {len(polygon_points)} điểm: {polygon_points}")
            else:
                print("Đã hủy vẽ polygon. Thoát ứng dụng.")
                return
        except Exception as e:
            print(f"Lỗi khi khởi tạo Polygon Drawer: {e}")
            return
    
    # Khởi tạo Controller và GUI
    controller = Controller(config)
    
    # Truyền polygon data vào stream thread nếu có
    if polygon_points:
        controller.stream_thread.set_polygon(polygon_points)
    
    window = MainWindow(controller, config)
    
    # Bắt đầu xử lý
    controller.start()
    
    # Hiển thị giao diện
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

