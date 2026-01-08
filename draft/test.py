import cv2
import time
from pathlib import Path
from sahi_t1 import SAHI

model_path = "D:/ATIN/SAHI_T1/last.pt"
sahi = SAHI(model_path, conf=0.5, iou=0.5, imgsz=1280)

print("=== SAHI Image Inference ===")
print("Nhập đường dẫn ảnh để infer")
print("Nhập 'q' hoặc để trống để thoát\n")

while True:
    image_path = input("Nhập đường dẫn ảnh: ").strip()

    if image_path == "" or image_path.lower() == "q":
        print("Thoát chương trình.")
        break

    if not Path(image_path).exists():
        print("Không tìm thấy ảnh\n")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh\n")
        continue

    t0 = time.time()
    boxes, scores, classes = sahi.predict(img, n_slices=3, overlap=0.2)
    t1 = time.time()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{int(cls)}:{score:.2f}", (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out_path ="D:/ATIN/SAHI_T1/result.jpg"
    cv2.imwrite(out_path, img)

    print(f"Detected {len(boxes)} objects")
    print(f"Predict time: {(t1 - t0) * 1000:.1f} ms")
    print(f"Saved -> {out_path}\n")
