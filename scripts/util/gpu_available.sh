#!/bin/bash

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    exit 255
else
    exit 0
fi
