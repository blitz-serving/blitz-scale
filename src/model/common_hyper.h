#pragma once

#include <unistd.h>

#include <unordered_map>

#include "hyper.h"

namespace blitz::model {

const GptHyperParam HYPERPARAM_OPT_125M =
    GptHyperParam::GetOptHyperParam(  // opt-125m
        50272,
        2048,
        768,
        12,
        12,
        64,
        3072
    );

const GptHyperParam HYPERPARAM_OPT_1P3B =
    GptHyperParam::GetOptHyperParam(  // opt-1.3b
        50272,
        2048,
        2048,
        24,
        32,
        64,
        8192
    );

const GptHyperParam HYPERPARAM_OPT_2P7B =
    GptHyperParam::GetOptHyperParam(  // opt-2.7b
        50272,
        2048,
        2560,
        32,
        32,
        80,
        10240
    );

const GptHyperParam HYPERPARAM_OPT_6P7B =
    GptHyperParam::GetOptHyperParam(  // opt-6.7b
        50272,
        2048,
        4096,
        32,
        32,
        128,
        16384
    );

const GptHyperParam HYPERPARAM_OPT_13B =
    GptHyperParam::GetOptHyperParam(  // opt-13b
        50272,
        2048,
        5120,
        40,
        40,
        128,
        20480
    );

const GptHyperParam HYPERPARAM_OPT_30B =
    GptHyperParam::GetOptHyperParam(  // opt-30b
        50272,
        2048,
        7168,
        48,
        56,
        128,
        28672
    );

const GptHyperParam HYPERPARAM_LLAMA2_7B =
    GptHyperParam::GetLlama2HyperParam(  // llama2-7b
        32000,
        4096,
        4096,
        32,
        32,
        32,
        128,
        11008
    );

const GptHyperParam HYPERRPARAM_LLAMA3_8B =
    GptHyperParam::GetLlama2HyperParam(  // llama3-8b
        128256,
        4096,
        4096,
        32,
        32,
        8,
        128,
        14336
    );

const GptHyperParam HYPERPARAM_LLAMA2_13B =
    GptHyperParam::GetLlama2HyperParam(  // llama2-13b
        32000,
        4096,
        5120,
        40,
        40,
        40,
        128,
        13824
    );

const GptHyperParam HYPERPARAM_LLAMA2_70B =
    GptHyperParam::GetLlama2HyperParam(  // llama2-70b
        32000,
        4096,
        8192,
        80,
        64,
        8,
        128,
        28672
    );

const GptHyperParam HYPERPARAM_MISTRAL_13B =
    GptHyperParam::GetLlama2HyperParam(  // Qwen2-14b
        32003,
        4096,
        4096,
        60,
        32,
        8,
        128,
        14336
    );

const GptHyperParam HYPERPARAM_MISTRAL_24B = GptHyperParam::GetLlama2HyperParam(
    131072,
    32768,
    5120,
    40,
    32,
    8,
    128,
    32768
);

const GptHyperParam HYPERPARAM_QWEN_72B = GptHyperParam::GetLlama2HyperParam(
    152064,
    32768,
    8192,
    80,
    64,
    8,
    128,
    29568
);

// str2hyperparam - Return the correct hyperparam based on the string.
// If the string is invalid, print the valid hyperparam and return a
// hyperparam with vocab_size = -1.
inline GptHyperParam str2hyperparam(const std::string& str) {
    static const std::unordered_map<std::string, GptHyperParam>
        hyper_param_map = {
            {"opt_125m", HYPERPARAM_OPT_125M},
            {"opt_1.3b", HYPERPARAM_OPT_1P3B},
            {"opt_2.7b", HYPERPARAM_OPT_2P7B},
            {"opt_6.7b", HYPERPARAM_OPT_6P7B},
            {"opt_13b", HYPERPARAM_OPT_13B},
            {"opt_30b", HYPERPARAM_OPT_30B},
            {"llama2_7b", HYPERPARAM_LLAMA2_7B},
            {"llama2_13b", HYPERPARAM_LLAMA2_13B},
            {"llama2_70b", HYPERPARAM_LLAMA2_70B},
            {"llama3_8b", HYPERRPARAM_LLAMA3_8B},
            {"mistral_13b", HYPERPARAM_MISTRAL_13B},
            {"mistral_24b", HYPERPARAM_MISTRAL_24B},
            {"qwen_72b", HYPERPARAM_QWEN_72B},
        };

    if (hyper_param_map.find(str) == hyper_param_map.end()) {
        printf("Invalid number of parameters: %s\n", str.c_str());
        printf("Valid number of parameters: ");
        for (auto it = hyper_param_map.begin(); it != hyper_param_map.end();
             ++it) {
            printf("%s ", it->first.c_str());
        }
        exit(1);
        GptHyperParam res = HYPERPARAM_OPT_125M;
        res.vocab_size = -1;
        return res;
    }

    return hyper_param_map.at(str);
}

}  // namespace blitz::model