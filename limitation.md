# 限界
- depth-anything が実空間のスケールと比例するdepthを返すという保証がない。
- この方式での補完処理では、zed-sdk での値と補完された値の間で食い違い、段差を生じる。
- その大きさが、抑えきれているとは言えない。
- zed-sdk の値をそのまま採用するという方針の妥当性の根拠がない。
- RANSACでinlierにならなかった画素は、zed-sdkの値が大きく実際の値と離れている可能性がある。
- しかし、そのことを利用していない。
### ZED SDK のFILL MODE
```.py:
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
```
### ZED SDK のNEURAL
```.py:
init_parameters = sl.InitParameters()
init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
```

## 代案
- ステレオ画像に対して、視差を計算する最新のアルゴリズムを試してみる。
- CVPRなどの学会で発表されているアルゴリズムでは、対応点が算出できていない点でも、視差が検出できている。
- 例：
  - https://github.com/gangweiX/IGEV