```
python3 depth_viewer.py -h              
usage: depth_viewer.py [-h] [--sec SEC] [--gray] [--jet] [--inferno] captured_dir

depth npy file viewer

positional arguments:
  captured_dir  captured directory by capture.py

optional arguments:
  -h, --help    show this help message and exit
  --sec SEC     wait sec

colormap:
  --gray        gray colormap
  --jet         jet colormap
  --inferno     inferno colormap
```


```
python3 capture.py -h
usage: capture.py [-h] [--input_svo_file INPUT_SVO_FILE] [--ip_address IP_ADDRESS] [--resolution RESOLUTION] [--confidence_threshold CONFIDENCE_THRESHOLD] [--outdir OUTDIR]

capture stereo pairs

optional arguments:
  -h, --help            show this help message and exit
  --input_svo_file INPUT_SVO_FILE
                        Path to an .svo file, if you want to replay it
  --ip_address IP_ADDRESS
                        IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup
  --resolution RESOLUTION
                        Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
  --confidence_threshold CONFIDENCE_THRESHOLD
                        depth confidence_threshold(0 ~ 100)
  --outdir OUTDIR       image pair output

```