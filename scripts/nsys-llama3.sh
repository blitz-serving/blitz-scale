#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export RUST_BACKTRACE="full"
export LOG_LEVEL="info"

nsys launch mpirun -n 8 --allow-run-as-root ./build/release/bin/run_server_disaggregative -T /nvme/workdir/wht/model/Llama-2-7b-hf/tokenizer.json -V '' --host localhost --port 50051 --model-path '' --model-name llama3_8b -P fp16 --config ./log/profile/llama3_8b/config-server.json --num-total-blocks 30000
