import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

GRPC_ADDR = "127.0.0.1:50051"

CREATE     = "vdss.VDSSService/CreateCollection"
UPSERT     = "vdss.VDSSService/BatchUpsert"
SEARCH     = "vdss.VDSSService/Search"
HEALTH     = "vdss.VDSSService/HealthCheck"
GET_VECTOR = "vdss.VDSSService/GetVector"
GET_COUNT  = "vdss.VDSSService/GetVectorCount"

# Enums you discovered (grpcurl accepts enum NAMES in JSON too)
# IndexDriver: FAISS
# IndexAlgorithm: HNSW
# StorageType: MEM (recommended for container)
# DistanceMetric: COSINE / EUCLIDEAN / DOT

app = FastAPI()

DELETE = "vdss.VDSSService/DeleteCollection"

@app.delete("/collections/{name}")
def delete_collection(name: str):
    return grpc_call(DELETE, {"collection_name": name})


def grpc_call(method: str, payload: Dict[str, Any]) -> Any:
    p = subprocess.run(
        ["grpcurl", "-plaintext", "-d", json.dumps(payload), GRPC_ADDR, method],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise HTTPException(status_code=500, detail=p.stderr.strip() or "grpcurl failed")

    out = p.stdout.strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


@app.get("/health")
def health():
    # call the real HealthCheck RPC
    return grpc_call(HEALTH, {})


class CreateCollectionReq(BaseModel):
    name: str
    dimension: int

    # optional tuning
    distance_metric: str = "COSINE"   # COSINE / EUCLIDEAN / DOT
    storage_type: str = "MEM"        # MEM / BTRIEVE_FILE / BTRIEVE_SPACE

    # HNSW config (optional but recommended)
    m: int = 16
    ef_construct: int = 200
    ef_search: int = 64

    # advanced
    config_json: str = ""


class UpsertReq(BaseModel):
    collection: str
    ids: List[Union[int, str]]                 # ints => u64_id, strings => uuid
    vectors: List[List[float]]
    payloads: Optional[List[Dict[str, Any]]] = None


class SearchReq(BaseModel):
    collection: str
    query: List[float]
    top_k: int = 5
    with_vector: bool = False
    with_payload: bool = True
    filter: Optional[Dict[str, Any]] = None    # will be json-stringified into filter_json


@app.post("/collections")
def create_collection(req: CreateCollectionReq):
    payload: Dict[str, Any] = {
        "collection_name": req.name,
        "config": {
            "index_driver": "FAISS",
            "index_algorithm": "HNSW",
            "storage_type": req.storage_type,
            "dimension": req.dimension,
            "distance_metric": req.distance_metric,
            "config_json": req.config_json,
            "hnsw_config": {
                "m": req.m,
                "ef_construct": req.ef_construct,
                "ef_search": req.ef_search,
            },
        },
    }
    return grpc_call(CREATE, payload)


@app.post("/upsert")
def batch_upsert(req: UpsertReq):
    if len(req.ids) != len(req.vectors):
        raise HTTPException(status_code=400, detail="ids and vectors must have same length")

    dim = len(req.vectors[0]) if req.vectors else 0
    for v in req.vectors:
        if len(v) != dim:
            raise HTTPException(status_code=400, detail="all vectors must have the same dimension")

    payload_objs = req.payloads or [{} for _ in req.ids]
    if len(payload_objs) != len(req.ids):
        raise HTTPException(status_code=400, detail="payloads must be same length as ids (or omitted)")

    vector_ids: List[Dict[str, Any]] = []
    for _id in req.ids:
        if isinstance(_id, int):
            vector_ids.append({"u64_id": _id})
        else:
            vector_ids.append({"uuid": str(_id)})

    vectors: List[Dict[str, Any]] = [
        {"data": v, "dimension": dim} for v in req.vectors
    ]

    payloads: List[Dict[str, Any]] = [
        {"json": json.dumps(p)} for p in payload_objs
    ]

    payload: Dict[str, Any] = {
        "collection_name": req.collection,
        "vector_ids": vector_ids,
        "vectors": vectors,
        "payloads": payloads,
    }
    return grpc_call(UPSERT, payload)


@app.get("/count/{collection}")
def get_vector_count(collection: str):
    """Return the number of vectors stored in a collection."""
    result = grpc_call(GET_COUNT, {"collection_name": collection})
    return {"collection": collection, "count": int(result.get("count", 0))}


@app.get("/scan/{collection}")
def scan_collection(collection: str, limit: int = 10000):
    """
    Retrieve all chunk texts from a collection by fetching vectors by sequential integer ID.
    Chunks are stored with IDs 0..N-1 during upsert. Used by Modal at training time
    so text data lives in Actian (not R2) and Modal reads from Actian.
    """
    count_result = grpc_call(GET_COUNT, {"collection_name": collection})
    total = min(int(count_result.get("count", 0)), limit)

    if total == 0:
        return {"collection": collection, "count": 0, "texts": []}

    def fetch_one(i: int):
        try:
            result = grpc_call(GET_VECTOR, {
                "collection_name": collection,
                "vector_id": {"u64_id": i},
            })
            payload_json = result.get("payload", {}).get("json", "{}")
            payload = json.loads(payload_json) if payload_json else {}
            return i, payload.get("text", "")
        except Exception:
            return i, ""

    results: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch_one, i): i for i in range(total)}
        for future in as_completed(futures):
            i, text = future.result()
            results[i] = text

    texts = [results[i] for i in range(total) if results.get(i)]
    return {"collection": collection, "count": len(texts), "texts": texts}


@app.post("/search")
def search(req: SearchReq):
    dim = len(req.query)

    payload: Dict[str, Any] = {
        "collection_name": req.collection,
        "query": {"data": req.query, "dimension": dim},
        "top_k": req.top_k,
        "with_vector": req.with_vector,
        "with_payload": req.with_payload,
    }

    # filter_json is optional string
    if req.filter is not None:
        payload["filter_json"] = json.dumps(req.filter)

    return grpc_call(SEARCH, payload)