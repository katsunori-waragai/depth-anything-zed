#!/bin/sh
echo scp -r weights $(logname)@$(hostname).local:$(pwd)/ > copyto_host.sh

