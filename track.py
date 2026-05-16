from ultralytics import YOLO

model_path = r"F:\A课程\计算机视觉\HWK2-2\runs\detect\VisDrone_Detect\yolov8n_cpu_test\weights\best.pt"
model = YOLO(model_path)

print("开始视频跟踪推理，请稍候...")
results = model.track(
    source="test_video.mp4",
    show=True,
    tracker="botsort.yaml",
    save=True
)

print("跟踪完成！视频已保存。")