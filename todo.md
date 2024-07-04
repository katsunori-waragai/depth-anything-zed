# todo
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
