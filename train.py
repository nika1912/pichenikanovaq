from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8n.pt')
    result = model.train(
        data= 'data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        name='cookie_detector',
        patience=50
    )

