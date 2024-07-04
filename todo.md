# todo
- depthを画像ではなく数値として受け取ること
- 以下は、数値データを画像として処理する方法
```commandline
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
```

- 数値としてのdepthを



