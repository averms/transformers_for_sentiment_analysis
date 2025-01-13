FROM registry.access.redhat.com/ubi9/ubi-micro:latest AS base

FROM registry.access.redhat.com/ubi9/ubi:latest AS builder
ARG UV_VER=0.5.18
ARG UV_HASH=1dbaeffc5cfac769f99700c0fc8c4ef4494a339720c6bf8b79367b1acd701b46

RUN mkdir /mnt/micro
COPY --from=base / /mnt/micro

ADD --checksum=sha256:${UV_HASH} \
    https://github.com/astral-sh/uv/releases/download/${UV_VER}/uv-x86_64-unknown-linux-gnu.tar.gz \
    /tmp/uv.tar.gz

RUN <<DONE
set -eu

dnf install -y --installroot /mnt/micro --releasever 9 --setopt install_weak_deps=False \
    python3.12

tar -C /mnt/micro/usr/local/bin --strip-components=1 -xf /tmp/uv.tar.gz

dnf clean all --installroot /mnt/micro
DONE

FROM scratch
COPY --from=builder /mnt/micro /
WORKDIR /app
COPY / /app
# RUN uv sync

CMD ["uv", "run", "src/preprocess_imdb.py"]
