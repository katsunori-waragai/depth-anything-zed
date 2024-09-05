from pathlib import Path

def main(args):
    captured_dir = Path(args.captured_dir)
    leftdir = captured_dir / "left"
    rightdir = captured_dir / "right"
    zeddepthdir = captured_dir / "zed-depth"

    left_images = sorted(leftdir.glob("*.png"))
    for p in sorted(zeddepthdir.glob("*.npy")):
        print(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="depth npy file viewer")
    parser.add_argument("captured_dir", help="captured directory by capture.py")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", help="gray colormap")
    group.add_argument("--jet", help="jet colormap")
    args = parser.parse_args()
    print(args)
