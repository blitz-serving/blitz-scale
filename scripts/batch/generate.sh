#!/bin/bash

python -m grpc_tools.protoc -Iproto/ --python_out=./scripts/batchv2 --grpc_python_out=./scripts/batchv2 generate.proto
