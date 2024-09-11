## Yolo系列模型onnx格式推理
### 简介
本仓库提供Yolo系列模型onnx格式推理通用代码示例，用户可根据业务需求对推理输出进行后处理。代码结构如下：
* yolo_det_onnx_predictor: 目标识别任务推理代码
* yolo_seg_onnx_predictor: 目标分割任务推理代码

相关依赖:
* onnxruntime-gpu
* opencv-python

onnx格式导出：https://docs.ultralytics.com/zh/integrations/onnx/

### 参考
* https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation

