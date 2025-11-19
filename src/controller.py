from PyQt5.QtCore import QObject
from src.capture_thread import ThreadCapture
from src.process_thread import ThreadProcess
from src.stream_thread import ThreadStream

class Controller(QObject):
    """
    Controller quản lý khởi tạo và kết nối các luồng.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Khởi tạo các luồng
        self.capture_thread = ThreadCapture(config)
        self.process_thread = ThreadProcess(config)
        self.stream_thread = ThreadStream(config)
        
        # Kết nối tín hiệu
        self._connect_signals()

    def _connect_signals(self):
        """Kết nối các tín hiệu giữa các luồng."""
        # Capture -> Process
        self.capture_thread.new_frame.connect(self.process_thread.add_frame)
        
        # Process -> Stream
        self.process_thread.processed_results.connect(self.stream_thread.display_frame)

    def start(self):
        """Bắt đầu tất cả các luồng."""
        self.capture_thread.start()
        self.process_thread.start()
        self.stream_thread.start()

    def stop(self):
        """Dừng tất cả các luồng."""
        self.capture_thread.stop()
        self.process_thread.stop()
        self.stream_thread.stop()
