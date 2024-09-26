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


### 表示の改善のするべきこと
- zed-sdkで値が求まっているpixel について、両者の相関関係を確認すること。
- 期待すること：　１次式の関係にあること。
## fitting残差の表示をすること
- それが何％の誤差になるのか
## depthを対数軸ではなく、linearの軸で算出すること。

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

## depth_anythingでの推論が実行できなかったときのエラー
エラーを表示しても、スクリプトは継続する。
```commandline
[07/03/2024-07:37:30] [TRT] [E] 1: [resizeRunner.cpp::execute::89] Error Code 1: Cuda Runtime (invalid resource handle)
```
TRT を利用しているコード側の以下の改変で解決した。
https://github.com/katsunori-waragai/depth-anything-zed/pull/16


# Depth-anything とステレオ計測との相容れない部分
- ステレオ計測:
  - ポスターがあったら、ポスターの貼られている平面あるいは曲面を返すのを期待する。
- Depth-anything:
  - ポスターがあったら、ポスターに写っている内容を奥行きがあると解釈して結果を返す。
  - そのため、絵に対しても奥行きを解釈することがある。
