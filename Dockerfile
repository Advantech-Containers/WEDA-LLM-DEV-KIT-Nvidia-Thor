ARG BASE_IMAGE=harbor.edgesync.cloud/weda-ai/weda-llm-base:1.0.0-thor
FROM ${BASE_IMAGE}

# Install basic dev tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    htop \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wise-edge/weda-llm-base

# Copy custom configurations, scripts, and entrypoint
COPY config/ /wise-edge/weda-llm-base/config/
COPY scripts/ /wise-edge/weda-llm-base/scripts/
COPY entrypoint.sh /wise-edge/weda-llm-base/entrypoint.sh

# Ensure execution permissions for scripts and entrypoint
RUN chmod +x /wise-edge/weda-llm-base/entrypoint.sh \
             /wise-edge/weda-llm-base/scripts/wise-bench

# Declare default environment variables
ENV WEDA_LLM_PORT=9000 \
    HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/models/moe-configs/tiktoken \
    TIKTOKEN_RS_CACHE_DIR=/models/moe-configs/tiktoken

# Expose model serving port
EXPOSE 9000

# Set entrypoint and default command
ENTRYPOINT ["/wise-edge/weda-llm-base/entrypoint.sh"]
CMD ["serve"]
