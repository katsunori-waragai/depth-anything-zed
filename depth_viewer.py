import argparse

parser = argparse.ArgumentParser(description="depth npy file viewer")
parser.add_argument("captured_dir", help="captured directory by capture.py")
group = parser.add_argument_group("colormap")
group.add_argument("--gray", help="gray colormap")
group.add_argument("--jet", help="jet colormap")
args = parser.parse_args()
print(args)
