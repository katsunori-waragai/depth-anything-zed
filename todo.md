# todo
- testの実装
- 同じ入力データで、異なる手法での評価を可能にすること
- 視差ベースでの１次式（切片＝０）での合わせこみをする。
- そうすることで、視差が大きな部分でのフィッティング精度が保たれる。
- また視差ベースなので、十分に遠方ではともに視差が０になるという条件を満たすはずである。
- 視差ベースのフィッティングにしたときに、視差が大きい領域での残差を小さくしたい。
- depth-anything は、disparityの絶対値を出せない。
- test用のデータの追加
  - depth_file = "data/zed_depth.npy"
  - rgb_file = "data/left.png"
- 出力先の指定
camerainfo.py　の利用

モジュールへの移動
depth_sensing_inv.py　のモジュール部分の移動

## もし、このモジュールを再利用しようと思う場合には
- 点群への変換は、Open3Dベースのものに置き換えること。
- ZED　SDK依存の部分とZED　SDK非依存の部分を分離すること。
