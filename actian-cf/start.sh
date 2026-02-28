#!/usr/bin/env bash
set -euo pipefail

GRPC_HOST="127.0.0.1"
GRPC_PORT="50051"

echo "[boot] starting Actian VDSS gRPC server..."
/usr/local/actianvectorai/bin/vdss-grpc-server &
ACTIAN_PID=$!

echo "[boot] waiting for TCP ${GRPC_HOST}:${GRPC_PORT}..."
for i in {1..200}; do
  if nc -z "${GRPC_HOST}" "${GRPC_PORT}" >/dev/null 2>&1; then
    echo "[boot] TCP port is open."
    break
  fi
  sleep 0.1
done

if ! nc -z "${GRPC_HOST}" "${GRPC_PORT}" >/dev/null 2>&1; then
  echo "[error] gRPC port never opened. PID=${ACTIAN_PID}"
  exit 1
fi

# Optional: also verify reflection works (now that grpcurl is real)
echo "[boot] verifying grpcurl can list services..."
/usr/local/bin/grpcurl -plaintext "${GRPC_HOST}:${GRPC_PORT}" list

echo "[boot] starting HTTP gateway on :8080"
exec uvicorn api:app --host 0.0.0.0 --port 8080