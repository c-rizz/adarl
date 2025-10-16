#!/bin/bash

echo "Current governor:"
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_gov*

echo "Setting CPU governor to performance"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo "New governor:"
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_gov*