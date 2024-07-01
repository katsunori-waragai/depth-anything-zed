#!/bin/sh
# Edit here for your environment
dst=$(cat host_location.txt)
scp -r weights ${dst}/
