#!/bin/bash
# phát sự kiện mới từ pipe.log trên VM (pipe, lỗi, mỗi 50 bước train)
S=$(dirname $0); seen=$S/seen.txt; touch $seen
while true; do
  out=$($S/rsh 'grep -E "^\[pipe|PIPE_EXIT|Error|error:|Traceback|\"step\": (50|100|150|200|250|300),|xong" pipe.log | tail -40' 2>/dev/null)
  if [ -z "$out" ] && ! $S/rsh 'echo alive' 2>/dev/null | grep -q alive; then echo "VM_UNREACHABLE $(date +%H:%M)"; fi
  echo "$out" | grep -v '^$' | while IFS= read -r l; do grep -qxF "$l" $seen || { echo "$l" >> $seen; echo "$l" | cut -c1-220; }; done
  echo "$out" | grep -q PIPE_EXIT && exit 0
  sleep 150
done
