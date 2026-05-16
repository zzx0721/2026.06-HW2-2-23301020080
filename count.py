import cv2
from ultralytics import YOLO

# 1. 加载模型
model_path = r"best.pt"
model = YOLO(model_path)

# 2. 读取测试视频
video_path = "test_video.avi"
cap = cv2.VideoCapture(video_path)

# 3. 获取原视频的长宽和帧率，用于初始化输出视频
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 设置输出视频 (使用 mp4v 编码生成 .mp4 文件)
out = cv2.VideoWriter('count_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# 4. 设定虚拟线位置 (竖线：画在画面宽度 50% 的中间位置)
LINE_X = int(w * 0.5)
counted_ids = set()  # 集合：用于记录已经越线的 ID，防止同一个物体在右侧晃悠被重复计数
total_count = 0

print("开始进行视频逐帧越线计数，请稍候 (由于是逐帧处理，需要运行几分钟)...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 在画面上画出一条红色的垂直基准线 (从顶部画到底部)
    cv2.line(frame, (LINE_X, 0), (LINE_X, h), (0, 0, 255), 3)

    # 运行多目标跟踪 (逐帧传入，防止内存溢出)
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

    # 确保当前帧检测到了目标并且分配了 Tracking ID
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            # 计算目标的中心点坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 在目标的中心位置画一个绿色的实心圆点，方便观察
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # 【核心越线判断】：如果目标中心点的 X 坐标大于虚拟线的 X 坐标(即跑到了线的右侧)
            # 并且这个目标的 ID 之前没有被记过数
            if center_x > LINE_X and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

    # 在视频画面左上角，用蓝色粗体显示实时计数值
    cv2.putText(frame, f"Total Count: {total_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

    # 将处理好(带有线、点、计数数字)的这一帧写入到输出视频中
    out.write(frame)

# 5. 释放资源并结束
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f"计数完成！最终有 {total_count} 个目标跨越了该线。")
print("结果呈现：请在当前文件夹下查看新生成的 'count_result.mp4' 视频。")