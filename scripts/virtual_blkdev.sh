#!/bin/bash
# 创建可控读取速度的虚拟块设备
# Usage: sudo ./virtual_blkdev.sh <size_mb> <read_speed_mbps>
# Example: sudo ./virtual_blkdev.sh 100 10

set -e

SIZE_MB=${1:?Usage: $0 <size_mb> <read_speed_mbps>}
READ_SPEED_MBPS=${2:?Usage: $0 <size_mb> <read_speed_mbps>}

BACKING_FILE="/tmp/vblk_backing"
CGROUP="/sys/fs/cgroup/blkio/vblk"

# 清理旧设备
losetup -D 2>/dev/null || true
rm -f "$BACKING_FILE"
rmdir "$CGROUP" 2>/dev/null || true

# 创建后端文件和 loop 设备
dd if=/dev/zero of="$BACKING_FILE" bs=1M count="$SIZE_MB" status=progress
LOOP_DEV=$(losetup -f --show "$BACKING_FILE")

# 获取设备号并设置 blkio 限制 (cgroup v1)
MAJOR=$((16#$(stat -c '%t' "$LOOP_DEV")))
MINOR=$((16#$(stat -c '%T' "$LOOP_DEV")))
mkdir -p "$CGROUP"
echo "$MAJOR:$MINOR $((READ_SPEED_MBPS * 1024 * 1024))" > "$CGROUP/blkio.throttle.read_bps_device"

echo ""
echo "Device: $LOOP_DEV (${SIZE_MB}MB, ${READ_SPEED_MBPS}MB/s read limit)"
echo "Test:   echo \$\$ > $CGROUP/cgroup.procs && dd if=$LOOP_DEV of=/dev/null bs=1M"
echo "Clean:  sudo losetup -d $LOOP_DEV && rm $BACKING_FILE && rmdir $CGROUP"
