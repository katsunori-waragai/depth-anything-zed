# todo
- 数値としてのdepthを点群データに変換して妥当性を確認しやすくすること
- 以下のissue を読むと点群データへの変換と可視化の例が記されている。
https://github.com/LiheYoung/Depth-Anything/issues/36

## fitting残差の表示をすること
- それが何％の誤差になるのか

## depthを対数軸ではなく、linearの軸で算出すること。

## linearの軸で算出した結果を点群に変換すること
```commandline
@dataclass
class Depth2Points:
    fx: float
    fy: float
    cx: float
    cy: float
    def cloud_points(self, depth):
        pass
        
```

## ZED SDK での撮影の解像度のモードを確認すること
それによって、焦点距離の情報が変わってくることを考慮すること。
