FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends tor ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --chown=debian-tor:debian-tor torrc /etc/tor/torrc
USER debian-tor
ENTRYPOINT ["tor", "-f", "/etc/tor/torrc"]
