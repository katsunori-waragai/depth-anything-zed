from util_depth_view import view_by_colormap, view3d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="depth npy file viewer")
    parser.add_argument("captured_dir", help="captured directory by capture.py")
    parser.add_argument("--sec", type=int, default=1, help="wait sec")
    parser.add_argument("--vmax", type=float, default=5000, help="max depth [mm]")
    parser.add_argument("--vmin", type=float, default=0, help="min depth [mm]")
    parser.add_argument("--disp3d", action="store_true", help="display 3D")
    group = parser.add_argument_group("colormap")
    group.add_argument("--gray", action="store_true", help="gray colormap")
    group.add_argument("--jet", action="store_true", help="jet colormap")
    group.add_argument("--inferno", action="store_true", help="inferno colormap")
    args = parser.parse_args()
    print(args)
    if args.disp3d:
        view3d(args)
    else:
        view_by_colormap(args)
