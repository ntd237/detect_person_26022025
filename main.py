import sys
from PyQt5.QtWidgets import QApplication
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
    
    # Khởi tạo Controller và GUI
    controller = Controller(config)
    window = MainWindow(controller, config)
    
    # Bắt đầu xử lý
    controller.start()
    
    # Hiển thị giao diện
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
