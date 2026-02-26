#!/bin/bash

arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno main.ino
