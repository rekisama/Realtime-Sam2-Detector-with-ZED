# sam2_receiver.py
import threading
import zmq
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_object_tracker

VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
SAM_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG = "./configs/samurai/sam2.1_hiera_b+.yaml"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

latest_frame = None
latest_mask = None
lock = threading.Lock()

def resize_mask(mask_tensor, target_shape):
    mask = torch.tensor(mask_tensor, device='cpu')
    mask = torch.nn.functional.interpolate(mask, size=target_shape, mode="bilinear", align_corners=False)
    return (mask > 0.0).numpy()

def sam_thread(sam):
    global latest_frame, latest_mask
    while True:
        if latest_frame is not None:
            lock.acquire()
            frame = latest_frame.copy()
            lock.release()
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16 if 'cuda' in DEVICE else torch.float32):
                out = sam.track_all_objects(img=frame)
                mask = resize_mask(out["pred_masks"], (VIDEO_HEIGHT, VIDEO_WIDTH))
                lock.acquire()
                latest_mask = mask
                lock.release()

def main():
    global latest_frame, latest_mask

    # 初始化 SAM2
    sam = build_sam2_object_tracker(
        num_objects=2,
        config_file=SAM_CONFIG,
        ckpt_path=SAM_CHECKPOINT,
        device=DEVICE,
        verbose=False
    )

    # 初始化 ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    # 获取第一帧用于选择 ROI
    print("等待第一帧...")
    data = socket.recv()
    frame = np.frombuffer(data, dtype=np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 4))[..., :3]
    bbox = cv2.selectROI("选择目标（回车确认）", frame)
    cv2.destroyAllWindows()
    x, y, w, h = bbox
    box = np.array([[[x, y], [x+w, y+h]]])
    with torch.inference_mode():
        sam.track_new_object(img=frame, box=box)

    # 启动 SAM 分割线程
    threading.Thread(target=sam_thread, args=(sam,), daemon=True).start()

    print("开始接收并显示")
    colors = [[255, 105, 180], [0, 255, 0], [255, 0, 0]]

    while True:
        data = socket.recv()
        frame = np.frombuffer(data, dtype=np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 4))[..., :3]

        lock.acquire()
        latest_frame = frame.copy()
        mask = latest_mask.copy() if latest_mask is not None else None
        lock.release()

        # 叠加分割遮罩
        vis_frame = frame.copy()
        if mask is not None:
            for i in range(min(mask.shape[0], len(colors))):
                obj_mask = mask[i, 0]
                vis_frame[obj_mask] = colors[i % len(colors)]

        cv2.imshow("SAM2 + ZED 实时分割", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    main()
