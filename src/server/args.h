#include <grpcpp/server.h>
#include <mpi.h>

#include <argparse/argparse.hpp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "generate.pb.h"
#include "include/handle_exception.h"
#include "model/common_hyper.h"

using namespace generate::v2;

/// \note: terminal input args
class Args {
  private:
    std::string model_path;
    std::string model_name;
    std::string tokenizer_json_path;
    std::string vocab_json_path;
    std::string precision;
    std::string host;
    std::string config_path;
    uint16_t port;
    uint16_t tp_degree;
    uint16_t pp_degree;
    bool debug;
    bool all_ready;
    uint64_t num_total_blocks;
    uint32_t ibv_rate;
    std::string log_dir;
    uint32_t time_to_live_in_ms;

  public:
    static Args parse_args(int argc, char* argv[]) {
        argparse::ArgumentParser program("run_gpt", "0.1");

        program.add_description(
            "This programs starts a gRPC server containing a model."
            "example: "
        );

        program.add_argument("--model-path").help("Path to the model weight file").required();
        program.add_argument("--model-name").help("Name of the model").required();
        program.add_argument("-T", "--tokenizer-json-path")
            .help("Path to the huggingface tokenizer json file")
            .required();
        program.add_argument("-V", "--vocab-json-path").help("Path to the vocab json file").required();
        program.add_argument("-P", "--precision").help("Precision used for inference (fp16 or fp32)").required();
        program.add_argument("-D", "--debug").help("Enable debug mode").default_value(false).implicit_value(true);
        program.add_argument("--host").help("Server listening host").default_value(std::string("localhost"));
        program.add_argument("--config").help("Initial replica states").default_value(std::string());
        program.add_argument("--port").help("Server listening port").default_value(std::string("50051"));
        program.add_argument("-PP", "--pipeline-parallelism")
            .help("Degree of pipeline parallelism")
            .default_value(std::string("1"));
        program.add_argument("-TP", "--tensor-parallelism")
            .help("Degree of tensor parallelism")
            .default_value(std::string("1"));
        program.add_argument("-A", "--all-ready")
            .help("2 groups of bijection are parameterised")
            .default_value(false)
            .implicit_value(true);
        program.add_argument("--num-total-blocks")
            .help("Number of total available blocks")
            .default_value(std::string("8000"));
        program.add_argument("--ibv-rate").help("IBV rate").default_value(std::string("0"));
        program.add_argument("--log-dir").help("Output log dir. Default to console").default_value(std::string(""));
        program.add_argument("--time-to-live").help("Time to live for model cache").default_value(std::string("100"));

        Args args;
        try {
            program.parse_args(argc, argv);
            args.model_path = program.get<std::string>("model-path");
            args.tokenizer_json_path = program.get<std::string>("tokenizer-json-path");
            args.vocab_json_path = program.get<std::string>("vocab-json-path");
            args.model_name = program.get<std::string>("model-name");
            args.precision = program.get<std::string>("precision");
            args.host = program.get("host");
            args.port = std::stoi(program.get("port"));
            args.config_path = program.get("config");
            args.pp_degree = std::stoi(program.get("pipeline-parallelism"));
            args.tp_degree = std::stoi(program.get("tensor-parallelism"));
            args.debug = program.get<bool>("debug");
            args.all_ready = program.get<bool>("all-ready");
            args.num_total_blocks = std::stoull(program.get("num-total-blocks"));
            args.ibv_rate = std::stoi(program.get("ibv-rate"));
            args.log_dir = program.get("log-dir");
            args.time_to_live_in_ms = std::stoi(program.get("time-to-live"));
        }
        EXIT_ON_EXCEPTION
        return args;
    }

    const std::string& get_model_path() const {
        return model_path;
    }

    const std::string& get_config_path() const {
        return config_path;
    }

    const std::string& get_model_name() const {
        return model_name;
    }

    const std::string& get_tokenizer_json_path() const {
        return tokenizer_json_path;
    }

    const std::string& get_vocab_json_path() const {
        return vocab_json_path;
    }

    const std::string& get_precision() const {
        return precision;
    }

    const std::string& get_host() const {
        return this->host;
    }

    const std::string& get_log_dir() const {
        return this->log_dir;
    }

    uint64_t get_num_total_blocks() const {
        return this->num_total_blocks;
    }

    uint16_t get_port() const {
        return this->port;
    }

    uint16_t get_tp_degree() const {
        return this->tp_degree;
    }

    uint16_t get_pp_degree() const {
        return this->pp_degree;
    }

    uint32_t get_ibv_rate() const {
        return this->ibv_rate;
    }

    bool is_debug() const {
        return debug;
    }

    bool is_all_ready() const {
        return all_ready;
    }

    uint32_t get_time_to_live_in_ms() const {
        return time_to_live_in_ms;
    }
};

/// \note: parse terminal input args
enum class Precision { FP32, FP16, INVALID };

const std::unordered_map<std::string, Precision> precision_map = {
    {"fp32", Precision::FP32},
    {"fp16", Precision::FP16},
    {"FP32", Precision::FP32},
    {"FP16", Precision::FP16},
    {"", Precision::INVALID}
};

Precision precision_from_string(const std::string& precision_str);
std::string precision_to_string(Precision precision);

struct RunArgs {
    std::string model_weight_path, tokenizer_json_path, vocab_json_path, input_path, listen_addr;
    ::blitz::model::GptHyperParam hyper_param;
    Precision precision;
    bool is_debug = false;

    RunArgs() = default;
    RunArgs(
        const std::string& model_weight_path,
        const std::string& tokenizer_json_path,
        const std::string& vocab_json_path,
        const std::string& input_path,
        const std::string& str_hyper_param,
        const std::string& str_precision,
        const std::string& listen_addr,
        bool is_debug = false
    );
};

/*-------------------IMPLEMENTATION------------------*/

inline Precision precision_from_string(const std::string& precision_str) {
    if (precision_map.find(precision_str) == precision_map.end()) {
        std::cerr << "Invalid precision string: " + precision_str << std::endl;
        return Precision::INVALID;
    }
    return precision_map.at(precision_str);
}

inline std::string precision_to_string(Precision precision) {
    for (auto it = precision_map.begin(); it != precision_map.end(); ++it) {
        if (it->second == precision) {
            return it->first;
        }
    }
    std::cerr << "Invalid precision: " + std::to_string(static_cast<int>(precision)) << std::endl;
    return "";
}

inline RunArgs::RunArgs(
    const std::string& model_weight_path,
    const std::string& tokenizer_json_path,
    const std::string& vocab_json_path,
    const std::string& input_path,
    const std::string& str_hyper_param,
    const std::string& str_precision,
    const std::string& listen_addr,
    bool is_debug
) :
    model_weight_path(model_weight_path),
    tokenizer_json_path(tokenizer_json_path),
    vocab_json_path(vocab_json_path),
    input_path(input_path),
    listen_addr(listen_addr),
    hyper_param(::blitz::model::str2hyperparam(str_hyper_param)),
    precision(precision_from_string(str_precision)),
    is_debug(is_debug) {}
