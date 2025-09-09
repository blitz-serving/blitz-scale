#include <fcntl.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "args.h"
#include "blitz/stub.h"
#include "blitz_tccl.h"
#include "handle_exception.h"
#include "include/logger.hpp"
#include "include/tokenizer.hpp"
#include "service/service.h"

using json = nlohmann::json;

namespace ds_config {
struct DisaggregationConfig {
    std::vector<std::string> init_states;
    std::vector<std::vector<int>> replicas;
    std::vector<int> machines;
};

void to_json(json& j, const DisaggregationConfig& p) {
    j = json {
        {"init_states", p.init_states},
        {"replicas", p.replicas},
        {"machines", p.machines}
    };
}

void from_json(const json& j, DisaggregationConfig& p) {
    j.at("init_states").get_to(p.init_states);
    j.at("replicas").get_to(p.replicas);
    j.at("machines").get_to(p.machines);
}

}  // namespace ds_config

// code for graceful shutdown
// static ptr to server
//! bug...
static std::unique_ptr<grpc::Server> server = nullptr;

static bool should_shutdown = false;
static std::thread shutdown_thread;
static std::mutex mutex;
static std::condition_variable cv;

static void shutdown_poller_main() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, []() { return should_shutdown; });
    server->Shutdown();
}

std::string* rank_str = nullptr;

template<typename DType, typename IdType>
int launch_disagg_servers(int argc, char* argv[]) {
    Args args;
    try {
        args = Args::parse_args(argc, argv);
    }
    EXIT_ON_EXCEPTION

    std::vector<int> prefill_ranks, decode_ranks, ready_ranks, inactive_ranks;
    std::vector<std::pair<int, int>> machine_rank_pairs;
    try {
        std::ifstream f(args.get_config_path());
        json data = json::parse(f);
        ds_config::DisaggregationConfig disaggregation_config =
            data.get<ds_config::DisaggregationConfig>();
        auto init_states = disaggregation_config.init_states;
        auto replicas = disaggregation_config.replicas;
        auto machines = disaggregation_config.machines;
        BZ_INFO("Machines: {}", ::fmt::join(machines, ","));
        assert(
            init_states.size() == replicas.size() * replicas[0].size()
        );
        assert(init_states.size() == machines.size());
        size_t nrank = init_states.size();
        size_t i = 0;
        for (const auto& replica : replicas) {
            std::string replica_state = init_states[i];
            for (int rank : replica) {
                auto init_state = init_states[i];
                assert((replica_state == init_state));
                if (init_state == "Prefill") {
                    prefill_ranks.push_back(rank);
                    ready_ranks.push_back(rank);
                } else if (init_state == "Decode") {
                    decode_ranks.push_back(rank);
                    ready_ranks.push_back(rank);
                } else if (init_state == "Normal") {
                    ready_ranks.push_back(rank);
                } else if (init_state == "Inactive") {
                    prefill_ranks.push_back(rank);
                    inactive_ranks.push_back(rank);
                } else {
                    throw ::fmt::format("Invalid init state: {}", init_state);
                }
                auto machine_id = machines[i];
                machine_rank_pairs.push_back({rank, machine_id});
                i++;
            }
        }
        assert((i == nrank));
    }
    EXIT_ON_EXCEPTION

    try {
        switch (args.get_ibv_rate()) {
            case 0:
                BlitzTccl::gRate = ibv_rate::IBV_RATE_MAX;
                break;
            case 100:
                BlitzTccl::gRate = ibv_rate::IBV_RATE_100_GBPS;
                break;
            case 200:
                BlitzTccl::gRate = ibv_rate::IBV_RATE_200_GBPS;
                break;
            default:
                throw std::runtime_error(
                    ::fmt::format("Invalid rate: {}", args.get_ibv_rate())
                );
        }
    }
    EXIT_ON_EXCEPTION

    RunArgs run_args;
    try {
        run_args = {
            args.get_model_path() + '/',
            args.get_tokenizer_json_path(),
            args.get_vocab_json_path(),
            "",
            args.get_model_name(),
            args.get_precision(),
            ""
        };
    }
    EXIT_ON_EXCEPTION

    // Init MPI
    int pvd = MPI_THREAD_MULTIPLE;
    // int stat = MPI_Init(&argc, &argv);
    int stat = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &pvd);
    if (stat != MPI_SUCCESS || pvd < MPI_THREAD_MULTIPLE) {
        BZ_ERROR("Failed to init MPI");
        return 1;
    }

    // Get the rank and size
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (args.get_log_dir() != "") {
        auto log_file =
            ::fmt::format("{}/rank-{}.log", args.get_log_dir(), rank);
        int out_file =
            open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        dup2(out_file, STDOUT_FILENO);
        dup2(out_file, STDERR_FILENO);
    }

    auto cuda_devices = BlitzTccl::parse_cuda_visible_devices();
    switch (cuda_devices[rank % cuda_devices.size()]) {
        case 0:
        case 1:
        case 2:
        case 3:
            BlitzTccl::bind_node(0);
            break;
        case 4:
        case 5:
        case 6:
        case 7:
            BlitzTccl::bind_node(1);
            break;
        default:
            throw std::runtime_error("Invalid rank");
    }

    BZ_INFO("Running with {} GPUs", world_size);
    BZ_INFO(
        "Ready ranks: {}; Inactive ranks: {}",
        ::fmt::join(ready_ranks, ","),
        ::fmt::join(inactive_ranks, ",")
    );
    rank_str = new std::string(std::to_string(rank));
    assert(
        (static_cast<size_t>(world_size)
         == ready_ranks.size() + inactive_ranks.size())
    );

    int tp_size = args.get_tp_degree(), pp_size = args.get_pp_degree();
    BZ_INFO(
        "rank={}; world_size={}; tp_size={}; pp_size={};",
        rank,
        world_size,
        tp_size,
        pp_size
    );

    /// \brief MPI Comm for each serving instance
    MPI_Comm inst_comm;
    int inst_rank, inst_size;
    {
        MPI_Comm_split(
            MPI_COMM_WORLD,
            rank / (tp_size * pp_size),
            rank,
            &inst_comm
        );
        MPI_Comm_rank(inst_comm, &inst_rank);
        MPI_Comm_size(inst_comm, &inst_size);
        assert((inst_size == tp_size * pp_size));
    }
    int tp_rank = 0;
    MPI_Comm tp_comm;
    // if (tp_size > 1) {
    // [0,1] [2,3] [4,5] [6,7]
    int tp_group = inst_rank / tp_size;
    MPI_Comm_split(inst_comm, tp_group, rank, &tp_comm);
    MPI_Comm_rank(tp_comm, &tp_rank);
    MPI_Comm_size(tp_comm, &tp_size);
    // }
    assert((pp_size == 1));
    int pp_rank = 0;
    // MPI_Comm pp_comm;
    // if (pp_size > 1) {
    // int pp_group = tp_rank;
    // MPI_Comm_split(grp_comm, pp_group, rank, &pp_comm);
    // MPI_Comm_rank(pp_comm, &pp_rank);
    // MPI_Comm_size(pp_comm, &pp_size);
    // }

    /// \brief MPI Comm for GPUs w/i one machine
    MPI_Comm machine_comm;
    assert(machine_rank_pairs[rank].first == rank);
    MPI_Comm_split(
        MPI_COMM_WORLD,
        machine_rank_pairs[rank].second,
        rank,
        &machine_comm
    );

    BZ_INFO(
        "Rank<{}>:(Instance Rank<{}>): TP size={}, rank={}; PP size={}, rank={}",
        rank,
        inst_rank,
        tp_size,
        tp_rank,
        pp_size,
        pp_rank
    );

    // Init parallel config
    blitz::model::GptParallelismParam
        parallel_config(tp_size, tp_rank, pp_size, pp_rank);
    parallel_config.init_by_hyper_param(run_args.hyper_param);
    auto& model_config = run_args.hyper_param;

    BlitzTccl::mpi_initialize_tccl_context(world_size, rank);

    // Init CUDA
    // int device = pp_rank * tp_size + tp_rank;
    int device = rank % cuda_devices.size();
    BZ_INFO("Rank<{}> set device {}", rank, device);
    cudaSetDevice(device);

    auto is_inactive =
        std::find(inactive_ranks.begin(), inactive_ranks.end(), rank)
        != inactive_ranks.end();
    BZ_INFO("Rank<{}> is inactive: {}", rank, is_inactive);

    BZ_INFO("Rank<{}> Num total blocks {}", rank, args.get_num_total_blocks())

    // clang-format off
    auto stub = blitz::Stub<DType, IdType>(
        !is_inactive,
        world_size,
        rank,
        device,
        nullptr,  // TODO: ib
        model_config.num_layers / args.get_pp_degree(),                     // layers
        model_config.num_q_heads,                                           // qo-head
        model_config.num_kv_heads ,                                         // kv-head
        model_config.head_dim,                                              // head-size
        model_config.hidden_size,                                           // hidden-size
        model_config.ffn_inter_dim,                                         // inter-size
        model_config.vocab_size,                                            // vocab
        1024,                                                               // max-batch-size
        8192,                                                               // max-batch-tokens
        64,                                                                 // page-size,
        args.get_num_total_blocks(),                                        // max-num-page
        machine_comm,                                                       // mpi machine comm
        tp_comm,                                                            // mpi tp comm
        args.get_tp_degree(),                                               // tp size
        tp_rank,                                                            // tp rank
        args.get_pp_degree(),                                               // pp size (hasn't supported yet)
        args.get_model_path(),
        args.get_model_name()
    );
    // clang-format on

    /// \note init communicator inside constructor
    // stub.init_communicator();
    blitz::HuggingfaceTokenizer encode_tokenizer =
        blitz::HuggingfaceTokenizer(args.get_tokenizer_json_path());

    // Init gRPC service
    std::vector<std::string> server_urls;
    auto host = args.get_host();
    auto port = args.get_port() + rank % cuda_devices.size();
    auto addr_uri = ::fmt::format("{}:{}", host, port);

    TextGenerationServiceImpl<DType, IdType> service = {
        world_size,
        rank,
        device,
        nullptr,  // TODO: ib
        std::unique_ptr<decltype(stub)>(&stub),
        run_args.hyper_param,
        std::move(encode_tokenizer),
        std::move(server_urls)
    };

    auto builder = ::grpc::ServerBuilder();
    builder.AddListeningPort(
        addr_uri,
        ::grpc::InsecureServerCredentials(),
        nullptr
    );
    builder.RegisterService(&service);
    server = builder.BuildAndStart();

    // code for gracefully shutdown
    auto handle = [](int s) {
        BZ_INFO("Rank<{}> Signal {} received, shutdown server", *rank_str, s);
        if (server)
            should_shutdown = true;
        cv.notify_one();
    };
    std::signal(SIGINT, handle);
    std::signal(SIGTERM, handle);
    std::signal(SIGQUIT, handle);

    BZ_INFO("Rank<{}> Start gRPC server on {}", rank, addr_uri);
    shutdown_thread = std::thread(shutdown_poller_main);
    server->Wait();

    return 0;
}

int main(int argc, char* argv[]) {
    launch_disagg_servers<half, int32_t>(argc, argv);
}