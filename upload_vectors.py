"""
Embed corpus chunks with the fine-tuned MiniLM model and upload all vectors
to Actian VectorAI DB running on localhost:50051.

Usage:
  python upload_vectors.py [--host localhost] [--port 50051]
                           [--collection corpus_embeddings]
                           [--model models/finetuned-MiniLM-corpus]
                           [--chunks corpus_chunks.json]
                           [--batch-size 256]

The script:
  1. Connects to the Actian VectorAI DB gRPC server.
  2. Creates (or re-opens) a collection configured for cosine similarity.
  3. Encodes all chunks in batches with the fine-tuned MiniLM model.
  4. BatchUpserts the vectors, storing the original text as JSON payload.
  5. Saves a snapshot and prints final stats.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import grpc
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Add generated stubs to path
sys.path.insert(0, str(Path(__file__).parent))
from vdss_client import (
    vdss_service_pb2 as svc,
    vdss_service_pb2_grpc as svc_grpc,
    vdss_types_pb2 as types,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_HOST       = "localhost"
DEFAULT_PORT       = 50051
DEFAULT_COLLECTION = "corpus_embeddings"
DEFAULT_MODEL      = "./models/finetuned-MiniLM-corpus"
DEFAULT_CHUNKS     = "./corpus_chunks.json"
DEFAULT_BATCH_SIZE = 256


# ── Embedding ─────────────────────────────────────────────────────────────────
def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    e = mask.unsqueeze(-1).expand(hidden.size()).float()
    return torch.sum(hidden * e, 1) / torch.clamp(e.sum(1), min=1e-9)


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 128,
) -> list[list[float]]:
    """Return L2-normalised embeddings as plain Python float lists."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    embs = F.normalize(embs, dim=-1)
    return embs.cpu().tolist()


# ── Actian VectorAI DB client helpers ─────────────────────────────────────────
def get_stub(host: str, port: int) -> svc_grpc.VDSSServiceStub:
    channel = grpc.insecure_channel(f"{host}:{port}")
    return svc_grpc.VDSSServiceStub(channel)


def health_check(stub: svc_grpc.VDSSServiceStub) -> str:
    resp = stub.HealthCheck(svc.HealthCheckRequest())
    return f"v{resp.version}  uptime={resp.uptime_seconds}s  code={resp.status.code}"


def ensure_collection(
    stub: svc_grpc.VDSSServiceStub,
    name: str,
    dimension: int,
) -> None:
    """Create the collection if it doesn't already exist."""
    cfg = types.CollectionConfig(
        index_driver=types.FAISS,
        index_algorithm=types.HNSW,
        storage_type=types.MEM,
        dimension=dimension,
        distance_metric=types.COSINE,
        hnsw_config=types.HnswConfig(m=16, ef_construct=200, ef_search=100),
    )
    resp = stub.CreateCollection(
        svc.CreateCollectionRequest(collection_name=name, config=cfg)
    )
    if resp.status.code == 0:
        print(f"  Collection '{name}' created (dim={dimension}, cosine, HNSW/FAISS).")
    else:
        # Code != 0 usually means it already exists — try opening it
        open_resp = stub.OpenCollection(svc.OpenCollectionRequest(collection_name=name))
        if open_resp.status.code == 0:
            print(f"  Collection '{name}' already exists — opened.")
        else:
            raise RuntimeError(
                f"Could not create or open collection '{name}': "
                f"{resp.status.message}"
            )


def batch_upsert(
    stub: svc_grpc.VDSSServiceStub,
    collection: str,
    chunk_batch: list[str],
    vector_batch: list[list[float]],
    id_offset: int,
) -> None:
    """Upload one batch of vectors with their text payloads."""
    vector_ids = [
        types.VectorIdentifier(u64_id=id_offset + i)
        for i in range(len(chunk_batch))
    ]
    vectors = [
        types.Vector(data=v, dimension=len(v))
        for v in vector_batch
    ]
    payloads = [
        types.Payload(json=json.dumps({"text": text, "id": id_offset + i}))
        for i, text in enumerate(chunk_batch)
    ]

    resp = stub.BatchUpsert(
        svc.BatchUpsertRequest(
            collection_name=collection,
            vector_ids=vector_ids,
            vectors=vectors,
            payloads=payloads,
        )
    )
    if resp.status.code != 0:
        raise RuntimeError(f"BatchUpsert failed: {resp.status.message}")


def get_stats(stub: svc_grpc.VDSSServiceStub, collection: str) -> dict:
    resp = stub.GetStats(svc.GetStatsRequest(collection_name=collection))
    s = resp.stats
    return {
        "total_vectors":      s.total_vectors,
        "indexed_vectors":    s.indexed_vectors,
        "storage_bytes":      s.storage_bytes,
        "index_memory_bytes": s.index_memory_bytes,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host",       default=DEFAULT_HOST)
    parser.add_argument("--port",       default=DEFAULT_PORT, type=int)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--chunks",     default=DEFAULT_CHUNKS)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int, dest="batch_size")
    args = parser.parse_args()

    # ── 1. Load chunks ────────────────────────────────────────────────────────
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}")
        sys.exit(1)
    chunks: list[str] = json.loads(chunks_path.read_text())
    print(f"Loaded {len(chunks)} chunks from '{chunks_path}'")

    # ── 2. Load fine-tuned embedding model ────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Loading model '{model_path}' on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModel.from_pretrained(str(model_path)).to(device)
    model.eval()

    # Probe embedding dimension
    probe = embed_texts(["probe"], tokenizer, model, device)
    dim = len(probe[0])
    print(f"Embedding dimension: {dim}")

    # ── 3. Connect to Actian VectorAI DB ─────────────────────────────────────
    print(f"\nConnecting to Actian VectorAI DB at {args.host}:{args.port} ...")
    stub = get_stub(args.host, args.port)
    print(f"  Health: {health_check(stub)}")

    # ── 4. Create / open collection ───────────────────────────────────────────
    ensure_collection(stub, args.collection, dim)

    # ── 5. Embed and upload in batches ────────────────────────────────────────
    total     = len(chunks)
    uploaded  = 0
    print(f"\nUploading {total} vectors (batch_size={args.batch_size}) ...")

    for start in range(0, total, args.batch_size):
        batch_texts   = chunks[start : start + args.batch_size]
        batch_vectors = embed_texts(batch_texts, tokenizer, model, device)
        batch_upsert(stub, args.collection, batch_texts, batch_vectors, id_offset=start)
        uploaded += len(batch_texts)
        pct = uploaded / total * 100
        print(f"  [{uploaded:>5}/{total}]  {pct:.1f}%")

    # ── 6. Flush + snapshot ───────────────────────────────────────────────────
    stub.Flush(svc.FlushRequest(collection_name=args.collection))
    stub.SaveSnapshot(svc.SaveSnapshotRequest(collection_name=args.collection))

    # ── 7. Final stats ────────────────────────────────────────────────────────
    stats = get_stats(stub, args.collection)
    print(f"\nDone!")
    print(f"  Collection:     {args.collection}")
    print(f"  Vectors stored: {stats['total_vectors']}")
    print(f"  Indexed:        {stats['indexed_vectors']}")
    print(f"  Storage:        {stats['storage_bytes'] / 1024:.1f} KB")
    print(f"  Index memory:   {stats['index_memory_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()
