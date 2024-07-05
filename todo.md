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

```commandline
Unable to cast Python instance to C++ type (compile in debug mode for details)
-------------------------------------------------------------------
PyCUDA ERROR: The context stack was not empty upon module cleanup.
-------------------------------------------------------------------
A context was still active when the context stack was being
cleaned up. At this point in our execution, CUDA may already
have been deinitialized, so there is no way we can finish
cleanly. The program will be aborted now.
Use Context.pop() to avoid this problem.
-------------------------------------------------------------------
Aborted (core dumped)
```
