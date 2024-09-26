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

