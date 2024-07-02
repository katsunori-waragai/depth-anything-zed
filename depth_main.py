import argparse

import lib_depth_engine
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--frame_rate', type=int, default=15, help='Frame rate of the camera')
    args.add_argument('--raw', action='store_true', help='Use only the raw depth map')
    args.add_argument('--stream', action='store_true', help='Stream the results')
    args.add_argument('--record', action='store_true', help='Record the results')
    args.add_argument('--save', action='store_true', help='Save the results')
    args.add_argument('--grayscale', action='store_true', help='Convert the depth map to grayscale')
    args = args.parse_args()

    lib_depth_engine.depth_run(args)
