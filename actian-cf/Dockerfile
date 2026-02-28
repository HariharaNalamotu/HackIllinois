FROM williamimoh/actian-vectorai-db:1.0b

WORKDIR /app

# Install Python + venv + curl + certs + netcat (for port readiness)
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        curl ca-certificates bash \
        netcat-openbsd tar; \
      rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
      apk add --no-cache \
        python3 py3-pip py3-virtualenv \
        curl ca-certificates bash \
        netcat-openbsd tar; \
    else \
      echo "No supported package manager (apt-get/apk) found."; \
      exit 1; \
    fi

# Create venv (avoids PEP 668 "externally managed environment")
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn pydantic

# ✅ Correct grpcurl install: download tarball + extract
RUN set -eux; \
    curl -fL -o /tmp/grpcurl.tar.gz \
      https://github.com/fullstorydev/grpcurl/releases/download/v1.9.2/grpcurl_1.9.2_linux_x86_64.tar.gz; \
    tar -xzf /tmp/grpcurl.tar.gz -C /usr/local/bin grpcurl; \
    chmod +x /usr/local/bin/grpcurl; \
    /usr/local/bin/grpcurl --version; \
    rm -f /tmp/grpcurl.tar.gz

COPY api.py /app/api.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]