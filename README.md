# BlitzScale

## Preface
Applogize for a "just-to-read" repo😭 This is maily because we lack a stable GPU supplyment last year (5/12 months totally available at school; some other times partially available, thx to Alibaba Tongyi). We have presented our work to five groups, academia and industry. The opinion of audience formed an overwhelming majority, that is, they want to see our code regardless it can one-button-run or not.

Fortunately, after OSDI'25 & ATC'25, our situation changes a lot, some vendors (thx to Alibaba Tongyi, Huawei Cloud, etc.) appreciate our work, and offer us resouces for refinement. We will fulfil ourselves to transform this "just-to-read" repo into a "easy-to-run" one **in one month**, for the sake of being a dedicated system researcher/programmer.

## Roadmap 

Please check our plan and process in the [pinned issue](https://github.com/blitz-serving/blitz-scale/issues/1)😘

## Compile flags for C++ (TBD)
We will provide 2 building methods sooner:
- A docker image with pre-compiled BlitzScale project
- A one-button-config script, which hiddens CMake details of FlashInfer, targeting Nvidia DGX systems now.

## Compile flags for Rust
Please read `router_v2/Cargo.toml`, our system is `impl_blitz,impl_fast_pro,impl_live_pro`, while our implemented baseline system is `impl_sllm,cache_replace`

## Docs (TBD)
We will provide docs in a story-telling way, once read, you can understand our full design!
Docs can be found on `https://blitz-serving.github.io/blitzscale-doc/`, but we are still working on it.

## Acknowledgements  
This project leverage awosome kernel libeary **FlashInfer**!
<https://github.com/flashinfer-ai/flashinfer>

This project incorporates code from **Text Generation Inference (TGI)**  
<https://github.com/huggingface/text-generation-inference>    
Copyright © 2022-present Hugging Face Inc.  
Licensed under the **Apache License, Version 2.0**.  
The full license text of apache-2.0 in TGI is provided in `LICENSES/Apache-2.0.txt`.  

The specification of modification to TGI file can be found in `NOTICE` and each source file.

This project is also inpired by **SwiftTransformer**,
<https://github.com/LLMServe/SwiftTransformer>
e.g., we learned that cuBLAS uses a columm-major storage format from their code.


