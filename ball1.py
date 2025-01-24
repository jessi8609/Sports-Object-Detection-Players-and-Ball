import os
from ultralytics import YOLO

def main():
    # Joriy ishchi katalogni ko'rsatish
    print("Current working directory:", os.getcwd())

    # Annotatsiya va datasetni tekshirish
    labels_path = "C:\\Users\\13\\Desktop\\ball\\train\\labels"
    if not os.path.exists(labels_path):
        print(f"Labels path does not exist: {labels_path}")
        return

    # Modelni yuklash va treningni boshlash
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="C:\\Users\\13\\Desktop\\ball\\data.yaml",
        epochs=50,
        batch=4,
        imgsz=640,
        workers=2,
        device="0",
        pretrained=True
    )

if __name__ == "__main__":
    main()
