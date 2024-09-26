# depth-anything-zed
docker environment for depth-anything　with ZED SDK

ZED SDK との連動を前提としたリポジトリです。

ZED SDK との連動を想定しない場合には、depth-anything-docker のリポジトリを用いてください。

## 参照元
Jetson 用のDepth-Anything のリポジトリ
https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin

#### original
https://github.com/LiheYoung/Depth-Anything

Project page
https://www.freemake.com/jp/free_video_downloader/

[Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/pdf/2401.10891)
CVPR 2024

## もくろみ
### ZED2iで深度が得られるのにdepth-anythingと組み合わせる理由
- ZED2iのステレオ計算が場合によって弱点を持っている。
- depth-anything はつじつまのあったdepthの解釈には長けているように見える。
##### ZED SDK単独で用いた場合の限界
- Depth Rangeの下限以下だと、欠損点になる。
- 左右のカメラで対応点がとれない箇所が、欠損点になる。
  - 特に左カメラでしか見えていない領域はsl.DEPTH_MODE.ULTRA では欠損値になる。
  - このため近接側の限界値を超えた近接側で、物体を見つけることができない。
- 透明物体に対する深度が、間違った値となる。
### depth-anything に期待すること
- 順序が安定しているdepth(深度情報)がとれること。
- 深度の絶対値は期待しない。
- segment-anything レベルでの解像度は期待しない。
- ZED SDK 環境でデータを取得できるので、depth_anything の結果とZED SDK でのdepthとを直接比較できる。
##### わかっていること
- Depth-Anythingの場合だと、近すぎる対象物でも距離が算出される。
- 遠すぎる対象物でも、それなりの値が算出される。欠損値とはならない。
##### 単眼depthが計算できる理由
- 照度差ステレオがある。
- 物体表面の輝度値をもとに物体表面の法線の向きを算出
- それらの情報を組み合せて物体の形状が算出できる。
- それら照度差ステレオでできていたことが、単眼deph計算が可能な裏付けになっている。
### 利用するRGB画像
- ステレオカメラでは、左画像の画素位置を基準に深度情報を計算するのが標準になっている。
- 単眼depth計算には、左カメラ画像を用いる。
### 改善したい内容
- ZED SDK のdepthのうち、近すぎてdepthが出ない領域を表示すること
- その領域に対してdepth-anything のdepthを表示させること。

## Install with Docker
- `python3 -m pip install .`
- 開発用のモジュールを含めるときには[dev]を追加することで[project.optional-dependencies]のdev =の内容がインストールされます。
- `python3 -m pip install .[dev]`
- インストール後の確認
```
python3 -m pip list | grep depanyzed
```

#### docker_build.sh
Dockerfileの中でモジュールのインストールをするように改変しました。
pythonが必要とするモジュールの記載はpyproject.toml に一元化してあります。


#### docker_run.sh
- host 環境のweights/ をguest環境の weights/ としてマウントするようにした。(./data/ も同様)
- そのため、guest環境でweight ファイルのダウンロードとTRTへの変換を一度行えば、2回目以降は利用できる。

#### model conversion from onnx model to TRT model
For the first time you have no TRT model files in ./weights/ directory.
Execute following command to convert onnx models to trt models.
```
python3 export_all_size.py
```
Be patient! 
This command takes 10 minutes or more.
And too many ignorable warnings.
After conversion
```commandline
 ls weights/
depth_anything_vits14_308.onnx  depth_anything_vits14_364.trt   depth_anything_vits14_518.onnx
depth_anything_vits14_308.trt   depth_anything_vits14_406.onnx  depth_anything_vits14_518.trt
depth_anything_vits14_364.onnx  depth_anything_vits14_406.trt
```


以下のdepth.py ではtensorRTに変換済みのモデルがあることを前提としている。
以下のコードでは、USBカメラを入力、元結果とdepth画像とを画面に表示する。


usb_depth_anything.py:  depth-anything using ZED2i as USB camera.
zed_depth_anything.py:   depth-anything using ZED2i with zed sdk.
zed_scaled_depth_anything.py: scaled depth-anything with help of disparity by zed sdk. 


## depth-anything with USB camera interface
The left and right images are retrieved from the cv2.VideoCapture() interface as one concatenated image.
ZED2i is used as a USB camera.
What you get is the disparity calculated from the left image only.
The disparity values are relative.

```commandline
 python3 usb_depth_anything.py -h
usage: usb_depth_anything.py [-h] [--frame_rate FRAME_RATE] [--raw] [--stream] [--record] [--save] [--grayscale]

depth-anything using zed2i as usb camera

optional arguments:
  -h, --help            show this help message and exit
  --frame_rate FRAME_RATE
                        Frame rate of the camera
  --raw                 Use only the raw depth map
  --stream              Stream the results
  --record              Record the results
  --save                Save the results
  --grayscale           Convert the depth map to grayscale


Script to calculate disparity with depth-anything from images acquired with a ZED2i camera using zed sdk
python3 zed_depth_anything.py -h
usage: zed_depth_anything.py [-h] [--input_svo_file INPUT_SVO_FILE] [--ip_address IP_ADDRESS] [--resolution RESOLUTION]
                             [--confidence_threshold CONFIDENCE_THRESHOLD]

depth-anything(native) with zed camera

optional arguments:
  -h, --help            show this help message and exit
  --input_svo_file INPUT_SVO_FILE
                        Path to an .svo file, if you want to replay it
  --ip_address IP_ADDRESS
                        IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup
  --resolution RESOLUTION
                        Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
  --confidence_threshold CONFIDENCE_THRESHOLD
                        depth confidence_threshold(0 ~ 100)

Script to acquire images from a ZED2i camera using zed sdk and calculate disparity with depth-anything with aligned scaling.
Robust fitting is performed so that the disparity converted from the depth of the stereo calculation in the zed sdk and the disparity in depth-anything are on the same straight line.
 
$ python3 zed_scaled_depth_anything.py -h
usage: zed_scaled_depth_anything.py [-h] [--quick] [--save_depth] [--save_ply] [--save_fullply]

scaled depth-anything

optional arguments:
  -h, --help      show this help message and exit
  --quick         simple output without matplotlib
  --save_depth    save depth and left image
  --save_ply      save ply
  --save_fullply  save full ply

```

- You can compare ZED SDK depth with scaled depth-anything as follows
```commandline
python3 zed_scaled_depth_anything.py --quick
```


## helper tools
- use disparity-view to capture and view npy files.
  - https://github.com/katsunori-waragai/disparity-view
  - zed_capture: capture tool 
  - disparity_viewer: disparity npy file viewer

## SEE ALSO
### depth-anything v2
https://github.com/DepthAnything/Depth-Anything-V2
[pdf](https://arxiv.org/abs/2406.09414)
[深度推定モデル Deep Anything v2を試してみる](https://qiita.com/d_sato_/items/2f6c553e771f1d05192e)


