# depth-anything-docker
docker environment for depth-anything

## 参照元
Jetson 用のDepth-Anything のリポジトリ
https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin

## もくろみ
- 順序が安定しているdepth(深度情報)がとれること。
- 深度の絶対値は期待しない。
- segment-anything レベルでの解像度は期待しない。

## docker 環境の外で
```commandline
bash gen_copy_script.sh
```
を実行しておく。
guest環境内weights/からhost環境のディレクトリをコピーするスクリプト
copyto_host.sh を生成させておく。
### 予め host 環境で `xhost +` を実行しておく

## docker_build.sh

## docker_run.sh

### モデルの変換(Docker環境内)
- ls weights
- モデルの変換を自動化する（onnx -> trt）
- export_all_size.py を追加した。
- 実行時間がかかる。
python3 export_all_size.py 

```commandline
 ls weights/
depth_anything_vits14_308.onnx  depth_anything_vits14_364.trt   depth_anything_vits14_518.onnx
depth_anything_vits14_308.trt   depth_anything_vits14_406.onnx  depth_anything_vits14_518.trt
depth_anything_vits14_364.onnx  depth_anything_vits14_406.trt
```

以下のdepth.py ではtensorRTに変換済みのモデルがあることを前提としている。
以下のコードでは、USBカメラを入力、元結果とdepth画像とを画面に表示する。

```commandline
python3 depth_main.py --stream

# as USB camera
python3 python3 zed_cam.py 

# use ZED SDK (not working)
python3 python3 zed_cam.py --use_zed_sdk
```
## host環境にtensorRTに変換後の重みファイルを保存しておくには
```commandline
bash copyto_host.sh
```

このようにして、weights/ ディレクトリの中身をhost環境に保存できる。

# TODO
- 他の方式でのDepthの推定と比較できるようにすること。
