import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from utils import *
from collocation_utils import *


scale_type = "ServerlessLLM"
usable_replica_num = 8
prefill_bar = 300
decode_bar = 100
thpt_calculated_by_cli = True
zoom_out_millis = None

events = ["up", "complete", "prefill down"]
collocation_events = ["up", "down", "relive"]


# latency-based
def plot_cli_data(
    cli_df, st, usable_replica_num, fig_path, cli_df2=None, scale_events=None
):
    if scale_events:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax3 = None

    plot_ttft(ax1, cli_df, "st")
    if scale_events is not None:
        plot_scale_events(
            events,
            scale_events,
            start_time=st,
            usable_replica_num=usable_replica_num,
            ax=ax1,
            ax2=ax3,
        )
    ax1.set_ylabel("latency (ms)")
    ax1.set_title(f"{scale_type} TTFT")
    ax1.legend()
    if ax3:
        ax3.set_title("Resource Usage")

    if cli_df2 is not None:
        plot_ttft(ax2, cli_df2, "distserve")
        ax2.set_xlabel("time (ms)")
        ax2.set_ylabel("latency (ms)")
        ax2.set_title("DistServer TTFT")
    else:
        # plot_max_tbt(ax2, cli_df)
        plot_avg_tbt(ax2, cli_df)
        # plot_p95_tbt(ax2, cli_df)
        ax2.set_xlabel("time (ms)")
        ax2.set_ylabel("latency (ms)")
        ax2.set_title(f"{scale_type} TBT")
        ax2.legend()

    plt.tight_layout()
    fig.savefig(fig_path)
    plt.close()
    # plt.show()


# throughput-based
def plot_thpt_data(st, thpt_metrics, scale_events, waiting_prefill, usable_replica_num):
    global events
    fig, (ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("throughpt token/s")
    ax1.set_title(f"{scale_type} Throughput")
    plot_thpts(ax1, thpt_metrics, start_time=st, case="prefill")
    plot_scale_events(
        events,
        scale_events,
        start_time=st,
        ax=ax1,
        ax2=ax2,
        usable_replica_num=usable_replica_num,
    )
    plot_thpts(ax3, thpt_metrics, start_time=st, case="decode")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # plot_waiting_prefill(ax, waiting_prefill, start_time=st)
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_slo_data(
    start_time,
    end_time,
    scale_events,
    cli_data,
    usable_replica_num,
    prefill_bar: float,
    decode_bar: float,
    fig_path: str,
    waiting_prefill=None,
    server_metrics=None,
):
    global events
    # first figure: prefill slo violation
    # second figure: decode slo violation
    # third figure: Resource Usage
    # fig, ax3 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

    # violated = cli_data[
    #     (cli_data["first_token_time"] > 3 * cli_data["calculation_time"])
    #     & (cli_data["first_token_time"] > 300)
    # ]

    violated = cli_data[
        (cli_data["first_token_time"] > cli_data["input_length"] * 0.375)
        & (cli_data["first_token_time"] > 300)
    ]

    seconds = pd.to_datetime(cli_data["s_time"], unit="ms").dt.floor("s")
    counts = seconds.value_counts().sort_index()

    counts_violated = (
        pd.to_datetime(violated["s_time"], unit="ms")
        .dt.floor("s")
        .value_counts()
        .sort_index()
    )
    ax0.bar(counts.index, counts.values, width=0.000005)
    ax0.bar(counts_violated.index, counts_violated.values, width=0.000005, color="red")
    ax0.set_title("Number of requests per Second")
    ax0.set_ylabel("Number of requests")

    # fig, (ax1, ax3) = plt.subplots(2,1,figsize=(12,6), sharex=True)
    plot_prefill_slo_violation(
        ax=ax1, client_df=cli_data, watermark=prefill_bar, scale_type=scale_type
    )
    # plot_scale_events(
    #     events=events,
    #     scale_events=scale_events,
    #     start_time=start_time,
    #     ax=ax1,
    #     ax2=None,
    #     usable_replica_num=None,
    # )
    # ax1.legend()
    if waiting_prefill:
        ax11 = ax1.twinx()
        plot_waiting_prefill(
            ax=ax11, waiting_prefill=waiting_prefill, start_time=start_time
        )
        ax11.legend(loc="lower left")
    elif server_metrics:
        ax11 = ax1.twinx()
        plot_thpts(
            ax=ax11,
            thpt_metrics=server_metrics,
            start_time=start_time,
            case="prefill",
            calculated_by_cli_data=thpt_calculated_by_cli,
            cli_data=cli_data,
        )
    plot_decode_slo_violation(
        ax=ax2,
        client_df=cli_data,
        key="max",
        watermark=decode_bar,
        scale_type=scale_type,
    )
    # print(f"Scale Events start time: {start_time}")
    plot_scale_events(
        events=events,
        scale_events=scale_events,
        start_time=start_time,
        end_time=end_time,
        ax=None,
        ax2=ax3,
        usable_replica_num=usable_replica_num,
    )
    ax3.set_title("Resource Usage")
    plt.tight_layout()
    fig.savefig(fig_path)
    plt.close()


def plot_collocatio_slo_data(
    start_time,
    scale_events,
    cli_data,
    usable_replica_num,
    prefill_bar: float,
    decode_bar: float,
    fig_path: str,
    waiting_prefill=None,
    server_metrics=None,
):
    global collocation_events
    # first figure: prefill slo violation
    # second figure: decode slo violation
    # third figure: Resource Usage

    # fig, ax3 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

    seconds = pd.to_datetime(cli_data["s_time"], unit="ms").dt.floor("s")
    counts = seconds.value_counts().sort_index()
    ax0.plot(counts.index, counts.values)
    ax0.set_title("Number of requests per Second")
    ax0.set_ylabel("Number of requests")

    # fig, (ax1, ax3) = plt.subplots(2,1,figsize=(12,6), sharex=True)
    plot_prefill_slo_violation(
        ax=ax1, client_df=cli_data, watermark=prefill_bar, scale_type=scale_type
    )
    # plot_scale_events(
    #     events=collocation_events,
    #     scale_events=scale_events,
    #     start_time=start_time,
    #     ax=ax1,
    #     ax2=None,
    #     usable_replica_num=None,
    # )
    # ax1.legend()

    if waiting_prefill:
        ax11 = ax1.twinx()
        plot_waiting_prefill(
            ax=ax11, waiting_prefill=waiting_prefill, start_time=start_time
        )
        ax11.legend(loc="lower left")
    elif server_metrics:
        ax11 = ax1.twinx()
        plot_thpts(
            ax=ax11,
            thpt_metrics=server_metrics,
            start_time=start_time,
            case="prefill",
            calculated_by_cli_data=thpt_calculated_by_cli,
            cli_data=cli_data,
        )
    plot_decode_slo_violation(
        ax=ax2,
        client_df=cli_data,
        key="avg",
        watermark=decode_bar,
        scale_type=scale_type,
    )
    plot_collocation_scale_events(
        events=collocation_events,
        scale_events=scale_events,
        start_time=start_time,
        ax=None,
        ax2=ax3,
        usable_replica_num=usable_replica_num,
    )
    ax3.set_title("Resource Usage")
    plt.tight_layout()
    fig.savefig(fig_path)
    plt.close()


def plot_main(router_log, client_log, fig_path):
    start_time, end_time, scale_events, metrics, waiting_prefill, p2d_migration, p2p_migration = (
        extract_log(router_log, zoom_out_millis=zoom_out_millis)
    )
    cli_data = prepare_data(client_log, zoom_out_millis=zoom_out_millis)
    # print(len(cli_data))
    plot_slo_data(
        start_time=start_time,
        end_time=end_time,
        scale_events=scale_events,
        cli_data=cli_data,
        usable_replica_num=usable_replica_num,
        prefill_bar=prefill_bar,
        decode_bar=decode_bar,
        fig_path=fig_path,
        waiting_prefill=None,
        server_metrics=None,
    )


def plot_collocation_main(router_log, client_log, fig_path):
    start_time, scale_events, metrics, waiting_prefill, p2d_migration, p2p_migration = (
        extract_collocation_log(router_log, zoom_out_millis=zoom_out_millis)
    )
    cli_data = prepare_data(client_log, zoom_out_millis=zoom_out_millis)
    plot_collocatio_slo_data(
        start_time=start_time,
        scale_events=scale_events,
        cli_data=cli_data,
        usable_replica_num=usable_replica_num,
        prefill_bar=prefill_bar,
        decode_bar=decode_bar,
        fig_path=fig_path,
        waiting_prefill=None,
        server_metrics=None,
    )


def plot_vllm_main(client_log, fig_path):
    cli_data = prepare_data(
        client_log, zoom_out_millis=zoom_out_millis, latency_scale=1000
    )
    plot_cli_data(
        cli_data,
        scale_events=None,
        st=0,
        usable_replica_num=8,
        fig_path=fig_path,
        cli_df2=None,
    )
