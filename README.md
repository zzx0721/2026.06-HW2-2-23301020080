# 2026.06-HW2-2-23301020080
计算机视觉Homework2任务二

# 场景目标检测与视频多目标跟踪 

本项目为计算机视觉课程作业，包含基于 YOLOv8 的单阶段目标检测模型微调训练、结合 BoT-SORT 的多目标跟踪（MOT），以及基于视频流的越线计数功能实现。

##  项目结构

* `train.py`：使用 VisDrone 无人机航拍数据集微调训练 YOLOv8n 模型的代码。
* `track.py`：加载自定义训练权重，利用 BoT-SORT 算法对测试视频进行逐帧多目标跟踪，并输出带有 Tracking ID 的视频。
* `count.py`：越线计数脚本。在画面水平中心设定垂直虚拟线，统计跨越该线的车辆/行人总数。
* `test_video.avi`：用于多目标跟踪与越线计数测试的原始输入视频。
* `count_result.mp4`：运行 `count.py` 后生成的带有计数结果的最终视频。

##  环境配置

本项目主要依赖 Ultralytics (YOLOv8)、OpenCV 和 wandb（用于可视化）。
请在 Python 终端中运行以下命令安装依赖：

```bash
pip install ultralytics wandb opencv-python lap
```

##  运行步骤
1. 模型训练
```Bash
python train.py
```

2. 多目标跟踪
```Bash
python track.py
```

3. 越线计数

```Bash
python count.py
```
