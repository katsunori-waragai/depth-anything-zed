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

## docker_build.sh
Dockerfileの中でモジュールのインストールをするように改変しました。
pythonが必要とするモジュールの記載はpyproject.toml に一元化してあります。

#### インストール
- `python3 -m pip install .`
- 開発用のモジュールを含めるときには[dev]を追加することで[project.optional-dependencies]のdev =の内容がインストールされます。
- `python3 -m pip install .[dev]`
- インストール後の確認
```
python3 -m pip list | grep depanyzed
```

## docker_run.sh
- host 環境のweights/ をguest環境の weights/ としてマウントするようにした。(./data/ も同様)
- そのため、guest環境でweight ファイルのダウンロードとTRTへの変換を一度行えば、2回目以降は利用できる。

#### whl ファイルを作りたいときは
[pyproject.toml をベースにwhl ファイルを作る手順](https://qiita.com/nonbiri15/items/5b517cf1b52e5ed6e1f9) を参照してください。


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


usb_depth_anything.py:  depth-anything using ZED2i as USB camera.
zed_capture.py:         capturing left right images using zed sdk.
zed_depth_anything.py:   depth-anything using ZED2i with zed sdk.
zed_scaled_depth_anything.py: scaled depth-anything with help of disparity by zed sdk. 


## USB カメラインタフェースでのdepth-anything
ZED SDK を使いません。
ZED2i はUSBカメラとして使います。
左画像、右画像は連結した１枚の画像としてcv2.VideoCapture() のインタフェースから取得します。
得られるものは、左画像だけから算出した視差(disparity)です。
視差の値は相対的なものです。
このままでは、単位のある深度に変換することができません。
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


ZED2iのカメラ入力を左画像・右画像として記録するためのものです。
root@waragai-orin:~/Depth-Anything-for-Jetson-Orin# python3 zed_capture.py -h       
usage: zed_capture.py [-h] [--input_svo_file INPUT_SVO_FILE] [--ip_address IP_ADDRESS] [--resolution RESOLUTION]
                      [--confidence_threshold CONFIDENCE_THRESHOLD] [--outdir OUTDIR]

capture stereo pairs

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
  --outdir OUTDIR       image pair output


ZED2i カメラをzed　sdkを使ってデータ取得して、depth-anything で視差を計算するスクリプト
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

ZED2i カメラをzed　sdkを使ってデータ取得して、スケーリングをそろえたdepth-anything で視差を計算するスクリプト
zed sdkでのステレオ計算の深度から変換した視差とdepth-anything の視差が同一の直線上になるようにロバストなフィッティングを実施している。
 python3 zed_scaled_depth_anything.py -h
usage: zed_scaled_depth_anything.py [-h] [--quick] [--save_depth] [--save_ply] [--save_fullply]

scaled depth-anything

optional arguments:
  -h, --help      show this help message and exit
  --quick         simple output without matplotlib
  --save_depth    save depth and left image
  --save_ply      save ply
  --save_fullply  save full ply

```

```commandline
python3 depth_main.py 

# use ZED SDK
python3 python3 zed_cam.py
```

# ZED-SDK でのdepthとdepth-anythingとの比較
```commandline
python3 depth_sensing.py

python3 depth_sensing.py --quick
```


### わかったこと
- log-log plotでのフィッティングよりも disparityベースでのフィッティングの方がよい。
- そのため、log-log　plotでのフィッティングは削除した。


### 表示の改善のするべきこと
- zed-sdkで値が求まっているpixel について、両者の相関関係を確認すること。
- 期待すること：　１次式の関係にあること。
## fitting残差の表示をすること
- それが何％の誤差になるのか
## depthを対数軸ではなく、linearの軸で算出すること。

## host環境にtensorRTに変換後の重みファイルを保存しておくには
weights ファイルがhost環境のディスク領域のmount にした。
そのため、なにもしなくても、次回のguest環境に引き継がれる。

# todo
- pointCloud への変換の確認方法
  - 球が球として計測できるか。
- スケーリングの妥当性を確認できているか？
- カメラ解像度と推論のための解像度の違いの扱いが妥当になっているか？
  - focal_length_x, focal_length_y の値との関連はどうか
- depth_anything を使ってdepthの値の絶対値を気にしている例はどれくらいあるのか？
- depth_anything での平面の平面性はどんなであるか
- 数値としてのdepthを点群データに変換して妥当性を確認しやすくすること
- 以下のissue を読むと点群データへの変換と可視化の例が記されている。
https://github.com/LiheYoung/Depth-Anything/issues/36


# troubleshooting
## そのzedデバイスで対応していないresolutionを指定してしまったときのエラー
```commandline
[2024-07-03 07:30:13 UTC][ZED][INFO] Logging level INFO
INVALID RESOLUTION
[2024-07-03 07:30:14 UTC][ZED][WARNING] INVALID RESOLUTION in sl::ERROR_CODE sl::Camera::open(sl::InitParameters)
[2024-07-03 07:30:14 UTC][ZED][ERROR] [ZED] sl::Camera::Open has not been called, no Camera instance running.
[2024-07-03 07:30:14 UTC][ZED][ERROR] [ZED] sl::Camera::Open has not been called, no Camera instance running.
```

## SEE ALSO
### depth-anything v2
https://github.com/DepthAnything/Depth-Anything-V2
[pdf](https://arxiv.org/abs/2406.09414)
[深度推定モデル Deep Anything v2を試してみる](https://qiita.com/d_sato_/items/2f6c553e771f1d05192e)

## ZED-SDK でのデータの取得と視差データの可視化には
- 以下のツールが作成してあります。
https://github.com/katsunori-waragai/disparity-view
- ２重でのメンテナンスを避けるため、このリポジトリで不要になったスクリプトは削除予定です。

## depth_anythingでの推論が実行できなかったときのエラー
エラーを表示しても、スクリプトは継続する。
```commandline
[07/03/2024-07:37:30] [TRT] [E] 1: [resizeRunner.cpp::execute::89] Error Code 1: Cuda Runtime (invalid resource handle)
```
TRT を利用しているコード側の以下の改変で解決した。
https://github.com/katsunori-waragai/depth-anything-zed/pull/16


## 将来的な入れ替えの可能性
- 単眼depthの分野の進展は継続中であり、それを考慮した実装にしたい。
- ステレオカメラでのdepth推定について軽量なライブラリがありうるので、単眼depthを使わないこともありうる。

## open3dの利用
- open3dで十分なことに対して、自前ライブラリを使わなくする。
- 

## ほしいもの
- [x] depth2anythingとzed-sdk とでのフィッティングの残差
- [x] マジックナンバーを減らすこと
- フィッティングを行うサンプリングを１フレームよりも増やすこと
  - そうすることで、対応点のとれる距離の範囲を広範囲にすること

## fitting　後の残差が大きくなりやすい領域は
- 推測
物体の輪郭に生じるartifact
細いことで、ブロックマッチングで対応がとれにくい領域
fittingの定義域の外
透明物体



# Depth-anything とステレオ計測との相容れない部分
- ステレオ計測:
  - ポスターがあったら、ポスターの貼られている平面あるいは曲面を返すのを期待する。
- Depth-anything:
  - ポスターがあったら、ポスターに写っている内容を奥行きがあると解釈して結果を返す。
  - そのため、絵に対しても奥行きを解釈することがある。

