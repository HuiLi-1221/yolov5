import argparse  # 命令行参数解析
from pathlib import Path  # 处理文件和路径

import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes,check_img_size, cv2
from utils.dataloaders import LoadImages, letterbox
from utils.plots import save_one_box


def detect(source, weights, output, imgsz=640, conf_thres=0.25, iou_thres=0.45):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查可用的设备CPU或GPU

    # Load model
    model = attempt_load(weights)  # 加载模型权重文件
    model.to(device).eval()  # 将模型放到设备上推理
    stride = int(model.stride.max())  # 获取模型步长
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    # Set Dataloader
    # 加载输入图像，接收输入图像文件夹的路径、图像尺寸、步长，并返回一个数据集对象
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    # 迭代加载数据集中的每张图像
    for path, img, im0s, _, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0-255 to 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]  # 把图像传递给模型进行推理，并得到预测结果 预测结果是一个列表 [0]表示提取列表第一个检测框  pred里保存了坐标、置信度、类别
        pred = non_max_suppression(pred, conf_thres, iou_thres)  # 使用非最大抑制方法对预测结果进行过滤，以除去重叠的边界框

        # Process detections
        for det in pred:
            if det is not None and len(det):  # 检查检测结果是否为空
                # 将边界框的坐标从归一化的值缩放回原始图像尺寸 将检测结果中的边界框的坐标从输入图像的尺寸转换为输出图像的尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()

                # Save images with bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    label = f'{[int(cls)]} {conf:.2f}'
                    xyxy = [int(x) for x in xyxy]  # Convert to integers
                    # 绘制边界框
                    cv2.rectangle(im0s, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                    # 添加标签
                    cv2.putText(im0s, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Save image
                # 将带有边界框的图像保存到指定的输出文件夹中
                save_path = str(Path(output) / Path(path).name)  # 创建保存路径
                cv2.imwrite(save_path, im0s)  # 保存图像


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建一个参数解析器对象
    parser.add_argument('--weights', type=str, default='best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='val/images', help='path to images directory')
    parser.add_argument('--output', type=str, default='output/exp', help='path to output directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    args = parser.parse_args()

    detect(args.source, args.weights, args.output, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
