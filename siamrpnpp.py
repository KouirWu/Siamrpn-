import argparse
import cv2 as cv
import numpy as np
import os

"""
原始论文链接 : https://arxiv.org/abs/1812.11703
原始仓库链接 : https://github.com/STVIR/pysot

您可以从以下链接下载跟踪器模型的预训练权重：
https://drive.google.com/file/d/11bwgPFVkps9AH2NOD1zBDdpF_tQghAB-/view?usp=sharing
您可以从以下链接下载 target net (SiamRPN++ 的目标分支)：
https://drive.google.com/file/d/1dw_Ne3UMcCnFsaD6xkZepwE4GEpqq7U_/view?usp=sharing
您可以从以下链接下载 search net (SiamRPN++ 的搜索分支)：
https://drive.google.com/file/d/1Lt4oE43ZSucJvze3Y-Z87CVDreO-Afwl/view?usp=sharing
您可以从以下链接下载 head model (RPN 头部)：
https://drive.google.com/file/d/1zT1yu12mtj3JQEkkfKFJWiZ71fJ-dQTi/view?usp=sharing
"""

class ModelBuilder():
    """ 
    此类用于构建 SiamRPN++ 跟踪器模型，它使用导入的 ONNX 网络。
    SiamRPN++ 主要由三个部分组成：
    1. target_net: 提取模板（目标）特征的主干网络。
    2. search_net: 提取搜索区域特征的主干网络。
    3. rpn_head: 区域建议网络（RPN）头部，用于分类和回归。
    """
    def __init__(self, target_net, search_net, rpn_head):
        super(ModelBuilder, self).__init__()
        # 构建目标分支（提取模板特征）
        self.target_net = target_net
        # 构建搜索分支（提取搜索区域特征）
        self.search_net = search_net
        # 构建 RPN 头部（进行相关性操作和预测）
        self.rpn_head = rpn_head

    def template(self, z):
        """ 
        以大小为 (1, 1, 127, 127) 的模板图像作为输入，生成卷积核（特征图）。
        这些特征图后续将在 RPN 头部中作为卷积核使用。
        """
        self.target_net.setInput(z)
        outNames = self.target_net.getUnconnectedOutLayersNames()
        # 前向传播，获取多层特征（SiamRPN++ 使用多层特征融合）
        self.zfs_1, self.zfs_2, self.zfs_3 = self.target_net.forward(outNames)

    def track(self, x):
        """ 
        以大小为 (1, 1, 255, 255) 的搜索区域图像作为输入，生成分类得分和边界框回归值。
        """
        self.search_net.setInput(x)
        outNames = self.search_net.getUnconnectedOutLayersNames()
        # 获取搜索区域的多层特征
        xfs_1, xfs_2, xfs_3 = self.search_net.forward(outNames)
        # 将模板特征作为输入传给 RPN 头部
        self.rpn_head.setInput(np.stack([self.zfs_1, self.zfs_2, self.zfs_3]), 'input_1')
        # 将搜索区域特征作为输入传给 RPN 头部
        self.rpn_head.setInput(np.stack([xfs_1, xfs_2, xfs_3]), 'input_2')
        outNames = self.rpn_head.getUnconnectedOutLayersNames()
        # RPN 头部前向传播，输出分类结果 (cls) 和回归结果 (loc)
        cls, loc = self.rpn_head.forward(outNames)
        return {'cls': cls, 'loc': loc}

class Anchors:
    """ 
    此类用于生成锚框 (Anchors)。
    锚框是预定义的框，用于在特征图的每个位置预测目标可能的位置和形状。
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        """
        根据预定义的步长、比例和尺度配置生成锚框。
        """
        anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride**2
        count = 0
        for r in self.ratios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                # 生成锚框的坐标 [x1, y1, x2, y2] -> 这里实际上是 [-w/2, -h/2, w/2, h/2] 
                # 即以 (0,0) 为中心的宽高
                anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1
        return anchors

class SiamRPNTracker:
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        # 锚框步长
        self.anchor_stride = 8
        # 锚框宽高比
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        # 锚框尺度
        self.anchor_scales = [8]
        # 跟踪基础大小
        self.track_base_size = 8
        # 上下文（context）数量，用于扩展搜索区域
        self.track_context_amount = 0.5
        # 模板图像大小
        self.track_exemplar_size = 127
        # 搜索实例图像大小
        self.track_instance_size = 255
        # 学习率，用于更新跟踪结果
        self.track_lr = 0.4
        # 惩罚因子，用于抑制尺度和长宽比剧烈变化
        self.track_penalty_k = 0.04
        # 窗口影响因子，用于抑制距离中心过远的目标
        self.track_window_influence = 0.44
        # 计算得分图的大小
        self.score_size = (self.track_instance_size - self.track_exemplar_size) // \
                          self.anchor_stride + 1 + self.track_base_size
        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)
        # 生成汉宁窗（Hanning Window），用于对远离中心的预测进行惩罚
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        # 生成所有位置的锚框
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        从原图中裁剪子窗口，并进行预处理（填充、缩放等）。
        
        参数:
            im:         BGR 格式的输入图像帧
            pos:        裁剪中心位置 (cx, cy)
            model_sz:   模型所需的输入大小（如 127 或 255）
            original_sz: 需要从原图裁剪的大小
            avg_chans:  通道均值，用于填充边界
        返回:
            im_patch:   处理后的子窗口图像，格式为 (1, 3, H, W)
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_h, im_w, im_d = im.shape
        c = (original_sz + 1) / 2
        cx, cy = pos
        # 计算裁剪区域的坐标
        context_xmin = np.floor(cx - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(cy - c + 0.5)
        context_ymax = context_ymin + sz - 1
        # 计算填充大小（如果裁剪区域超出图像边界）
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_w + 1))
        bottom_pad = int(max(0., context_ymax - im_h + 1))
        
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            # 如果需要填充，创建一个新的填充后的图像
            size = (im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_d)
            te_im = np.zeros(size, np.uint8)
            # 将原图放入中间位置
            te_im[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = im
            # 使用通道均值填充边界
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + im_w, :] = avg_chans
            if bottom_pad:
                te_im[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, im_w + left_pad:, :] = avg_chans
            # 裁剪所需的区域
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            # 如果不需要填充，直接裁剪
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        # 如果裁剪大小与模型所需大小不一致，进行缩放
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv.resize(im_patch, (model_sz, model_sz))
        # 调整维度顺序为 (Channels, Height, Width)
        im_patch = im_patch.transpose(2, 0, 1)
        # 增加 batch 维度
        im_patch = im_patch[np.newaxis, :, :, :]
        # 转换为 float32
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def generate_anchor(self, score_size):
        """
        生成用于特征图的锚框。
        这些锚框会平铺到整个得分图（score map）上。
        
        参数:
            score_size: 特征图（得分图）的大小
        返回:
            anchor:     所有位置的锚框坐标 [x, y, w, h]
        """
        anchors = Anchors(self.anchor_stride, self.anchor_ratios, self.anchor_scales)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        # 将锚框转换为 [cx, cy, w, h] 格式
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchors.anchor_num
        # 将基础锚框平铺到 score_size * score_size 的网格上
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        # 计算网格的起始偏移量
        ori = - (score_size // 2) * total_stride
        # 生成网格坐标
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        # 将网格坐标加到锚框中心上，得到每个位置的锚框
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        """
        将网络预测的边界框回归值转换为实际的边界框坐标。
        
        参数:
            delta:      网络输出的回归值 (4, H, W, A) -> 实际上这里处理后的形状不同
            anchor:     对应的锚框 [cx, cy, w, h]
        返回:
            delta:      预测的边界框 [cx, cy, w, h]
        """
        # 调整维度顺序并展平
        delta_transpose = np.transpose(delta, (1, 2, 3, 0))
        delta_contig = np.ascontiguousarray(delta_transpose)
        delta = delta_contig.reshape(4, -1)
        # 根据 RPN 回归公式计算预测框
        # pred_cx = delta_x * anchor_w + anchor_cx
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        # pred_cy = delta_y * anchor_h + anchor_cy
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        # pred_w = exp(delta_w) * anchor_w
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        # pred_h = exp(delta_h) * anchor_h
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _softmax(self, x):
        """
        在通道维度上进行 Softmax 操作，计算分类概率。
        """
        x = x.astype(dtype=np.float32)
        x_max = x.max(axis=1)[:, np.newaxis]
        e_x = np.exp(x-x_max)
        div = np.sum(e_x, axis=1)[:, np.newaxis]
        y = e_x / div
        return y

    def _convert_score(self, score):
        """
        将网络输出的分类得分转换为前景概率。
        
        参数:
            score:      网络输出的分类得分
        返回:
            score:      前景类的概率得分
        """
        # 调整维度顺序 (2, H, W, A) -> (H, W, A, 2)
        score_transpose = np.transpose(score, (1, 2, 3, 0))
        score_con = np.ascontiguousarray(score_transpose)
        # 展平为 (2, N)
        score_view = score_con.reshape(2, -1)
        score = np.transpose(score_view, (1, 0))
        # 计算 Softmax
        score = self._softmax(score)
        # 返回前景类的概率（索引 1）
        return score[:,1]

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """
        将边界框限制在图像范围内。
        """
        bbox_h, bbox_w = boundary
        cx = max(0, min(cx, bbox_w))
        cy = max(0, min(cy, bbox_h))
        width = max(10, min(width, bbox_w))
        height = max(10, min(height, bbox_h))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        初始化跟踪器。
        
        参数:
            img(np.ndarray):    BGR 格式的初始帧图像
            bbox: (x, y, w, h): 初始目标边界框
        """
        x, y, w, h = bbox
        # 初始化目标中心位置
        self.center_pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
        self.h = h
        self.w = w
        # 计算包含上下文的宽高
        w_z = self.w + self.track_context_amount * np.add(h, w)
        h_z = self.h + self.track_context_amount * np.add(h, w)
        # 计算搜索区域大小（基于正方形假设）
        s_z = round(np.sqrt(w_z * h_z))
        # 计算图像通道均值
        self.channel_average = np.mean(img, axis=(0, 1))
        # 裁剪并处理模板图像
        z_crop = self.get_subwindow(img, self.center_pos, self.track_exemplar_size, s_z, self.channel_average)
        # 将模板输入模型进行初始化（提取模板特征）
        self.model.template(z_crop)

    def track(self, img):
        """
        在当前帧中跟踪目标。
        
        参数:
            img(np.ndarray): BGR 格式的当前帧图像
        返回:
            bbox(list):[x, y, width, height] 预测的目标边界框
        """
        # 计算搜索区域大小（包含上下文）
        w_z = self.w + self.track_context_amount * np.add(self.w, self.h)
        h_z = self.h + self.track_context_amount * np.add(self.w, self.h)
        s_z = np.sqrt(w_z * h_z)
        # 计算缩放比例：模型输入大小 / 实际搜索区域大小
        scale_z = self.track_exemplar_size / s_z
        # 计算搜索实例的大小（在原图尺度上）
        s_x = s_z * (self.track_instance_size / self.track_exemplar_size)
        # 裁剪并处理搜索区域图像
        x_crop = self.get_subwindow(img, self.center_pos, self.track_instance_size, round(s_x), self.channel_average)
        # 运行模型进行跟踪，获取分类得分和回归值
        outputs = self.model.track(x_crop)
        # 处理分类得分
        score = self._convert_score(outputs['cls'])
        # 处理回归值，得到预测框
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # 计算尺度变化惩罚
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.w * scale_z, self.h * scale_z)))

        # 计算长宽比变化惩罚
        r_c = change((self.w / self.h) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        # 综合惩罚因子
        penalty = np.exp(-(r_c * s_c - 1) * self.track_penalty_k)
        # 应用惩罚到得分
        pscore = penalty * score

        # 应用窗口惩罚（抑制距离中心过远的目标）
        pscore = pscore * (1 - self.track_window_influence) + \
                 self.window * self.track_window_influence
        # 找到得分最高的锚框索引
        best_idx = np.argmax(pscore)
        # 将预测框还原到原图尺度
        bbox = pred_bbox[:, best_idx] / scale_z
        # 更新学习率（结合惩罚和得分）
        lr = penalty[best_idx] * score[best_idx] * self.track_lr

        cpx, cpy = self.center_pos
        x,y,w,h = bbox
        # 计算预测框中心在原图中的绝对坐标
        cx = x + cpx
        cy = y + cpy

        # 平滑边界框大小（使用滑动平均）
        width = self.w * (1 - lr) + w * lr
        height = self.h * (1 - lr) + h * lr

        # 裁剪边界框，防止超出图像边界
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # 更新跟踪器状态
        self.center_pos = np.array([cx, cy])
        self.w = width
        self.h = height
        # 转换返回格式为 [x, y, w, h]
        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        return {'bbox': bbox, 'best_score': best_score}

def get_frames(video_name):
    """
    视频帧生成器。
    
    参数:
        video_name: 视频文件路径。如果为 None，则使用摄像头。
    返回:
        frame: 逐帧返回图像
    """
    cap = cv.VideoCapture(video_name if video_name else 0)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def main():
    """ 
    SiamRPN++ 跟踪器主函数
    """
    # OpenCV DNN 支持的计算后端
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    # 计算目标设备
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
               cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(description='使用此脚本运行 SiamRPN++ 视觉跟踪器',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_video', type=str, help='输入视频文件的路径。跳过此参数以从摄像头捕获帧。')
    parser.add_argument('--target_net', type=str, default='target_net.onnx', help='目标分支（模板）的 ONNX 模型路径。')
    parser.add_argument('--search_net', type=str, default='search_net.onnx', help='搜索分支的 ONNX 模型路径。')
    parser.add_argument('--rpn_head', type=str, default='rpn_head.onnx', help='RPN 头部 ONNX 模型路径。')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="选择计算后端: "
                        "%d: 自动 (默认), "
                        "%d: Halide, "
                        "%d: Intel's Deep Learning Inference Engine (OpenVINO), "
                        "%d: OpenCV 实现, "
                        "%d: VKCOM, "
                        "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='选择目标设备: '
                        '%d: CPU (默认), '
                        '%d: OpenCL, '
                        '%d: OpenCL FP16, '
                        '%d: Myriad, '
                        '%d: Vulkan, '
                        '%d: CUDA, '
                        '%d: CUDA fp16 (半精度浮点)' % targets)
    args, _ = parser.parse_known_args()

    # 检查输入文件是否存在
    if args.input_video and not os.path.isfile(args.input_video):
        raise OSError("输入视频文件不存在")
    if not os.path.isfile(args.target_net):
        raise OSError("Target Net 文件不存在")
    if not os.path.isfile(args.search_net):
        raise OSError("Search Net 文件不存在")
    if not os.path.isfile(args.rpn_head):
        raise OSError("RPN Head Net 文件不存在")

    # 加载网络模型
    target_net = cv.dnn.readNetFromONNX(args.target_net)
    target_net.setPreferableBackend(args.backend)
    target_net.setPreferableTarget(args.target)
    search_net = cv.dnn.readNetFromONNX(args.search_net)
    search_net.setPreferableBackend(args.backend)
    search_net.setPreferableTarget(args.target)
    rpn_head = cv.dnn.readNetFromONNX(args.rpn_head)
    rpn_head.setPreferableBackend(args.backend)
    rpn_head.setPreferableTarget(args.target)
    
    # 构建模型和跟踪器
    model = ModelBuilder(target_net, search_net, rpn_head)
    tracker = SiamRPNTracker(model)

    first_frame = True
    cv.namedWindow('SiamRPN++ Tracker', cv.WINDOW_AUTOSIZE)
    for frame in get_frames(args.input_video):
        if first_frame:
            try:
                # 第一帧：让用户选择感兴趣区域 (ROI)
                init_rect = cv.selectROI('SiamRPN++ Tracker', frame, False, False)
            except:
                exit()
            # 初始化跟踪器
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            # 后续帧：执行跟踪
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            x,y,w,h = bbox
            # 绘制边界框
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.imshow('SiamRPN++ Tracker', frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

if __name__ == '__main__':
    main()
