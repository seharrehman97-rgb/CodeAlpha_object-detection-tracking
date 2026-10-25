Benchmark on data/street.mp4, first 150 frames, width 960, CPU only

| Configuration | Avg FPS | Detect (ms) | Track (ms) | Unique IDs |
|---|---|---|---|---|
| Baseline (YOLO11n, 640, histogram) | 6.47 | 101 | 13 | 19 |
| Smaller input (YOLO11n, 480, histogram) | 9.07 | 82 | 15 | 14 |
| Bigger model (YOLO11s, 640, histogram) | 4.49 | 190 | 19 | 34 |
| CNN embedder (YOLO11n, 640, mobilenet) | 1.33 | 265 | 479 | 20 |
