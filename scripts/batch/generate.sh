#!/bin/bash

python -m grpc_tools.protoc -Iproto/ --python_out=./scripts/batch --grpc_python_out=./scripts/batch \
    proto/common.proto proto/engine.proto proto/resource.proto proto/admin.proto
