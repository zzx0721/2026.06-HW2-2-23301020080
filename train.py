from ultralytics import YOLO
import wandb

wandb.login()

model = YOLO('yolov8n.pt')

print("开始在 CPU 上进行训练，请耐心等待...")
results = model.train(
    data='VisDrone.yaml',
    epochs=3,             # 受于硬件限制，只跑 3 轮
    imgsz=640,
    batch=4,
    device='cpu',
    lr0=0.01,
    optimizer='auto',
    project='VisDrone_Detect',
    name='yolov8n_cpu_test'
)
print("训练完成！")