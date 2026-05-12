#!/bin/bash

REMOTE_USER="danny"
REMOTE_IP="192.168.1.18"

REMOTE_BASE="/opt/poky/3.1.31/sysroots/aarch64-poky-linux"
LOCAL_BASE="/opt/poky/3.1.31/sysroots/aarch64-poky-linux"

echo "===> Sync libraries (remote -> local)"
sudo rsync -avz --progress \
    $REMOTE_USER@$REMOTE_IP:$REMOTE_BASE/usr/lib64/libopencv* \
    $LOCAL_BASE/usr/lib64/

echo "===> Sync include"
sudo rsync -avz --progress \
    $REMOTE_USER@$REMOTE_IP:$REMOTE_BASE/usr/include/opencv4 \
    $LOCAL_BASE/usr/include/

echo "===> Sync pkgconfig"
sudo rsync -avz \
    $REMOTE_USER@$REMOTE_IP:$REMOTE_BASE/usr/lib64/pkgconfig/opencv4.pc \
    $LOCAL_BASE/usr/lib64/pkgconfig/

echo "===> Sync cmake config"
sudo rsync -avz \
    $REMOTE_USER@$REMOTE_IP:$REMOTE_BASE/usr/lib64/cmake/opencv4 \
    $LOCAL_BASE/usr/lib64/cmake/

echo "===> Verification (compare file list)"

ssh $REMOTE_USER@$REMOTE_IP "find $REMOTE_BASE/usr/lib64 -name 'libopencv*' | sort" > remote_libs.txt
find $LOCAL_BASE/usr/lib64 -name "libopencv*" | sort > local_libs.txt

echo "Missing in local:"
comm -23 remote_libs.txt local_libs.txt

echo "Extra in local:"
comm -13 remote_libs.txt local_libs.txt

echo "Done."