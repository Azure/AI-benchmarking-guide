#!/bin/bash
set -euo pipefail
printf "%4s %7s %7s\n" "CPU" "NODE0" "NODE1"
 
for cpu in $(seq 0 64 127)
do
printf "%4s " ${cpu}
for numa in $(seq 0 1)
do
result=$(numactl -C ${cpu} -m ${numa} ../../multichase/multichase -s 512 -m 1g -n 120)
printf "%7.1f " ${result}
done
printf "\n"
done