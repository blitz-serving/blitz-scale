FROM nvcr.io/nvidia/pytorch:24.10-py3 AS grpc-builder

RUN apt update && apt install -y cmake build-essential autoconf pkg-config libssl-dev
COPY ./grpc /root/grpc
WORKDIR /root/grpc
RUN mkdir -p cmake/build && \
    cd cmake/build && \
    cmake \
        -DgRPC_BUILD_TESTS=OFF \
        -DgRPC_INSTALL=ON \
        -DCMAKE_INSTALL_PREFIX=/root/.local \
        ../.. && \
    make -j64 && \
    make install

FROM nvcr.io/nvidia/pytorch:24.10-py3

COPY --from=grpc-builder /root/.local /root/.local
WORKDIR /root
RUN apt update && \
    apt install -y vim wget curl git cmake build-essential autoconf pkg-config libssl-dev openssh-server && \
    apt reinstall libibverbs-dev
RUN echo 'export CMAKE_PREFIX_PATH=/root/.local' >> /root/.bashrc
RUN echo 'export PATH=/root/.local/bin:$PATH' >> /root/.bashrc

CMD ["/bin/bash"]