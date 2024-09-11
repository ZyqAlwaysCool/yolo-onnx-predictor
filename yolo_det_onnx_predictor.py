import onnxruntime
import cv2
import numpy as np

class YoloDetOnnxPredictor(object):
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        self.initialize_model()
        
    def initialize_model(self):
        #session_options = onnxruntime.SessionOptions() #新版本onnx
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] #旧版本onnx
        try:
            #self.session = onnxruntime.InferenceSession(self.model_path, session_options) #新版本onnx
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers) #旧版本onnx
            print("Running on GPU.")
        except:
            #self.session = onnxruntime.InferenceSession(self.model_path, session_options) #新版本onnx
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers) #旧版本onnx
            print("CUDA is not available. Running on CPU")
        
        self.get_input_details()
        self.get_output_details()
    
    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        return self.process_output(outputs)
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor
    
    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        #print('output.shape=({})'.format(outputs[0].shape))
        
        return outputs
    
    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        
        print('prediction.shape=({})'.format(predictions.shape))
        
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        boxes = self.extract_boxes(predictions)
        
        indices = self.nms(boxes, scores, self.iou_threshold)
        cap = np.random.rand(len(boxes), 5)
        cap[:, 0:4] = boxes.astype('int')
        cap[:, 4] = class_ids
        results = []
        for i in indices:
            results.append(cap[i].tolist())
        
        return results
    
    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        
        boxes = self.rescale_boxes(boxes)
        
        boxes = self.xywh2xyxy(boxes)
        
        return boxes
    
    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2]/2
        y[..., 1] = x[..., 1] - x[..., 3]/2
        y[..., 2] = x[..., 0] + x[..., 2]/2
        y[..., 3] = x[..., 1] + x[..., 3]/2
        return y
    
    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            
            keep_indices = np.where(ious < iou_threshold)[0]
            
            sorted_indices = sorted_indices[keep_indices + 1]
        
        return keep_boxes
    
    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):
        unique_class_ids = np.unique(class_ids)
        
        keep_boxes = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            
            class_keep_boxes = self.nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])
        
        return keep_boxes
    
    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        
        iou = intersection_area / union_area
        
        return iou

if __name__ == '__main__':
    model_path = '/path/to/your/model.onnx'
    predictor = YoloDetOnnxPredictor(model_path)
    
    img = cv2.imread('path/to/your/image.jpg')
    result = predictor.detect_objects(img)