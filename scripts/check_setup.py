import cv2
import numpy as np
import torch
import ultralytics

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("Ultralytics version:", ultralytics.__version__)
print("GPU available:", torch.cuda.is_available())
