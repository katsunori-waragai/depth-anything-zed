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
- 順序が安定しているdepth(深度情報)がとれること。
- 深度の絶対値は期待しない。
- segment-anything レベルでの解像度は期待しない。
- ZED SDK 環境でデータを取得できるので、depth_anything の結果とZED SDK でのdepthとを直接比較できる。
### ZED SDK単独で用いた場合の限界
- Depth Rangeの下限以下だと、欠損点になる。
- 左右のカメラで対応点がとれない箇所が、欠損点になる。
- 透明物体に対する深度が、間違った値となる。
### 単眼depthが計算できる理由
- 照度差ステレオがある。
- 物体表面の輝度値をもとに物体表面の法線の向きを算出
- それらの情報を組み合せて物体の形状が算出できる。
- それら照度差ステレオでできていたことが、単眼deph計算が可能な裏付けになっている。
### 利用するRGB画像
- ステレオカメラでは、左画像の画素位置を基準に深度情報を計算するのが標準になっている。
- 単眼depth計算には、左カメラ画像を用いる。
## わかっていること
- Depth-Anythingの場合だと、近すぎる対象物でも距離が算出される。
- 遠すぎる対象物でも、それなりの値が算出される。欠損値とはならない。
### 改善したい内容
- ZED SDK のdepthのうち、近すぎてdepthが出ない領域を表示すること
- その領域に対してdepth-anything のdepthを表示させること。

### 予め host 環境で `xhost +` を実行しておく

## docker_build.sh

## docker_run.sh
- host 環境のweights/ をguest環境の weights/ としてマウントするようにした。
- そのため、guest環境でweight ファイルのダウンロードとTRTへの変換を一度行えば、2回目以降は利用できる。
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
python3 depth_main.py 

# use ZED SDK
python3 python3 zed_cam.py
```

# ZED-SDK でのdepthとdepth-anythingとの比較
```commandline
python3 depth_sensing.py

python3 depth_sensing.py --quick
```



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


## depth_anythingでの推論が実行できなかったときのエラー
エラーを表示しても、スクリプトは継続する。
```commandline
[07/03/2024-07:37:30] [TRT] [E] 1: [resizeRunner.cpp::execute::89] Error Code 1: Cuda Runtime (invalid resource handle)
```
TRT を利用しているコード側の以下の改変で解決した。
https://github.com/katsunori-waragai/depth-anything-zed/pull/16


## 将来的な入れ替えの可能性
- 単眼depthの分野の進展は継続中であり、それを考慮した実装にしたい。