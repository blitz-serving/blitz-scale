# BlitzScale

## Preface
Apologize for a "just-to-read" repo currently 😭. This is mainly because we lacked a stable GPU supply last year (only 5/12 months available at school),
with special thanks to Alibaba Tongyi Lab and Huawei Cloud for supporting our project.
Thus, we are still pushing on to transform this "just-to-read" repo into an "easy-to-run" one **in one month**. Stay tuned!

## Roadmap 

Please check our plan and process in the [pinned issue](https://github.com/blitz-serving/blitz-scale/issues/1)😘

## Compile C++ part engine (WIP)
Currently, you can build C++ part of our project step-by-step:
1. **Prerequisite**: gRPC is installed in your system. If not, you can install it manually as follow
    ```bash
        git clone --branch v1.66.0 --recurse-submodules https://github.com/grpc/grpc.git
        cd grpc && mkdir -p cmake/build
        cd cmake/build 
        cmake \
            -DgRPC_BUILD_TESTS=OFF \
            -DgRPC_INSTALL=ON \
            -DCMAKE_INSTALL_PREFIX=/root/.local \
            ../..
        make -j64
        make install
    ```
    Reinstall the Infiniband package, something wrong with Nvidia's container
    ```bash
        apt reinstall libibverbs-dev
        apt install autoconf pkg-config libssl-dev
    ```
2. clone with all submodules equipped, don't forget to use `git submodule update --init --recursive`
3. Config cmake and build. The following works within Nvidia's official pytorch container, and gRPC installed in `/root/.local`
    ```bash
        mkdir -p build/release
        export CUDA_HOME=<your-cuda-home>
        cmake -B build/release -DBUILD_MODE=release -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch -DCMAKE_CUDA_ARCHITECTURES=80 -DFLASHINFER_CUDA_ARCHITECTURES=80 -DTORCH_CUDA_ARCH_LIST='8.0'
        cmake --build build/release -j128
    ```
    If you're using Hopper GPU, please update CUDA arch to 90/9.0. However, since all the evaluations are conducted on Ampere GPU, we haven't calibrated the manually wrapped Flashinfer kernels on Hopper class.

Finally, we will provide 2 building methods:
- A docker image with pre-compiled BlitzScale project
- A one-button-config script, which hides CMake details of FlashInfer, targeting Nvidia DGX systems now.

## Compile Rust part orchestrator
Stay at the project root directory, and use command `cargo build --release` to build all components. Afterwards, you can use `cargo build -p router_v2 --features <impl>` to turn on/off different implementations. 

Please read the comments in `router_v2/Cargo.toml` for more details. Our system is `impl_blitz,impl_fast_pro,impl_live_pro`, while our implemented baseline system is `impl_sllm,cache_replace`

## Run the whole system (WIP)
The whole BlitScale system consists of multiple C++ backend as workers and one Rust frontend as orchestrator to route request and trigger autoscaling. For test and evaluation, we also provide a trace reproducer acting as client named `request-sim`.

To run the 3 components together for various evaluation settings, we develop a batch job runner in `scripts/batch`. The batch runner takes a batch job description toml file named *template*, and instantiates all the toml config file of each run, named *instance*. According each *instance* file, batch runner will recompile the router part according to enabled features, and fill the client with designated trace. The `config` directory still remains some test/evaluation configurations when we develop this project. You need to change the IP where C++ workers listen on, and the NIC name according to your machine.

## Known issues
The models are unable to generate correct tokens. This is due to the fast-changing kernels of Flashinfer, a lack of implementing document, and its diminishing support on native C++ APIs. To follow the trend, we do consider engage some python code to keep pace with the most powerful LLM kernels.

## Docs (TBD)
We will provide docs in a story-telling way, once read, you can understand our full design!
Docs can be found on `https://blitz-serving.github.io/blitzscale-doc/`, but we are still working on it.

## Acknowledgements  
This project leverage awesome kernel library **FlashInfer**!
<https://github.com/flashinfer-ai/flashinfer>

This project incorporates code from **Text Generation Inference (TGI)**  
<https://github.com/huggingface/text-generation-inference>    
Copyright © 2022-present Hugging Face Inc.  
Licensed under the **Apache License, Version 2.0**.  
The full license text of apache-2.0 in TGI is provided in `LICENSES/Apache-2.0.txt`.  

The specification of modification to TGI file can be found in `NOTICE` and each source file.

This project is also inspired by **SwiftTransformer**,
<https://github.com/LLMServe/SwiftTransformer>
e.g., we learned that cuBLAS uses a column-major storage format from their code.


