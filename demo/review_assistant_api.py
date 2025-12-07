# app.py
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, PositiveInt

from inference_engine_V2 import DemoConfig, RevUtilEngine
import uvicorn

# ----------------------------
# Config 
# ----------------------------
LLM_BASE_URL = "http://10.127.105.10:8000/v1"    
LLM_MODEL = "k-chirkunov/RevUtil_merged_model"
MAX_BATCH = 128
MAX_TEXT_CHARS = 20000
MAX_CONCURRENT_WORKERS = 4  # Limit concurrent job processing
JOB_TTL_SECONDS = 24 * 60 * 60  # 1 day in seconds
CLEANUP_INTERVAL_SECONDS = 60 * 60  # Run cleanup every hour

# In-memory job store (job_id -> job dict)
JOBS: Dict[str, Dict[str, Any]] = {}


async def _cleanup_expired_jobs():
    """Remove jobs older than JOB_TTL_SECONDS."""
    current_time = time.time()
    expired_job_ids = [
        job_id
        for job_id, job in JOBS.items()
        if current_time - job.get("created_at", 0) > JOB_TTL_SECONDS
    ]
    for job_id in expired_job_ids:
        JOBS.pop(job_id, None)
    if expired_job_ids:
        print(f"Cleaned up {len(expired_job_ids)} expired job(s)")


async def _cleanup_worker():
    """Background task that periodically cleans up expired jobs."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        await _cleanup_expired_jobs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: start cleanup task on startup."""
    # Start cleanup task
    cleanup_task = asyncio.create_task(_cleanup_worker())
    yield
    # Cleanup on shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Revas Review Assistant API",
    root_path="/get_comments",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Worker pool semaphore to limit concurrent job processing
# Initialized lazily on first use (FastAPI ensures event loop exists)
_worker_semaphore: Optional[asyncio.Semaphore] = None


def _get_worker_semaphore() -> asyncio.Semaphore:
    """Get or create the worker semaphore."""
    global _worker_semaphore
    if _worker_semaphore is None:
        try:
            _worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
        except RuntimeError:
            # If no event loop exists, create a new one (shouldn't happen in FastAPI)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
    return _worker_semaphore


# ----------------------------
# Schemas
# ----------------------------
class GenerateParams(BaseModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = Field(default=4096)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    points: List[str] = Field(..., description="Array of review point strings to analyze")
    params: GenerateParams = Field(default_factory=GenerateParams)


class AspectResult(BaseModel):
    score: Optional[str] = None
    rationale: Optional[str] = None


class ItemResult(BaseModel):
    index: int
    text: str
    aspects: Dict[str, AspectResult]
    error: Optional[str] = None


class BatchResponse(BaseModel):
    request_id: str
    model: str
    results: List[ItemResult]
    latency_ms: int


JobStatus = Literal["queued", "running", "completed", "failed"]


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    response: Optional[BatchResponse] = None
    error: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def _create_engine(params: GenerateParams) -> RevUtilEngine:
    """Create a new RevUtil engine instance for a worker.
    
    Each worker gets its own engine instance to avoid:
    - Parameter conflicts (different jobs may have different temperature/max_tokens)
    - Thread-safety issues with shared OpenAI client
    - Resource contention
    """
    config = DemoConfig(
        api_base=LLM_BASE_URL,
        model_name=LLM_MODEL,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_tokens,
    )
    return RevUtilEngine(config=config, mock=False)


def _validate(req: BatchRequest) -> None:
    if not req.points:
        raise HTTPException(400, "points must be non-empty")
    if len(req.points) > MAX_BATCH:
        raise HTTPException(400, f"Too many points: {len(req.points)} > {MAX_BATCH}")
    for i, s in enumerate(req.points):
        if len(s) > MAX_TEXT_CHARS:
            raise HTTPException(400, f"Point[{i}] too long: {len(s)} chars > {MAX_TEXT_CHARS}")


async def _run_batch(req: BatchRequest, request_id: str) -> BatchResponse:
    """Run analyze_points on the batch of review points.
    
    Each worker creates its own engine instance for isolation and parameter independence.
    """
    t0 = time.perf_counter()

    try:
        # Create a new engine instance for this worker
        engine = _create_engine(req.params)
        # Run analyze_points (this is synchronous, but we're in an async context)
        # We'll run it in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        point_results = await loop.run_in_executor(None, engine.analyze_points, req.points)
        
        # Convert to ItemResult format
        results: List[ItemResult] = []
        for result in point_results:
            # Convert aspects dict to AspectResult objects
            aspects_dict: Dict[str, AspectResult] = {}
            for aspect_name, aspect_data in result.get("aspects", {}).items():
                aspects_dict[aspect_name] = AspectResult(
                    score=aspect_data.get("score"),
                    rationale=aspect_data.get("rationale"),
                )
            
            results.append(
                ItemResult(
                    index=result["index"],
                    text=result["text"],
                    aspects=aspects_dict,
                )
            )
        
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return BatchResponse(
            request_id=request_id,
            model=LLM_MODEL,
            results=results,
            latency_ms=latency_ms,
        )
    except Exception as e:
        # If batch fails, return error for all items
        results = [
            ItemResult(
                index=i,
                text=point,
                aspects={},
                error=str(e),
            )
            for i, point in enumerate(req.points, start=1)
        ]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return BatchResponse(
            request_id=request_id,
            model=LLM_MODEL,
            results=results,
            latency_ms=latency_ms,
        )


async def _job_worker(job_id: str, req: BatchRequest) -> None:
    """Process a single job. Uses semaphore to limit concurrent workers."""
    semaphore = _get_worker_semaphore()
    async with semaphore:
        # mark running
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["updated_at"] = time.time()

        try:
            request_id = str(uuid.uuid4())
            resp = await _run_batch(req, request_id=request_id)
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["response"] = resp.model_dump()
            JOBS[job_id]["error"] = None
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["response"] = None
        finally:
            JOBS[job_id]["updated_at"] = time.time()


# ----------------------------
# API
# ----------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/v1/health")
def health():
    return {"ok": True}


@app.post("/v1/jobs", response_model=JobCreateResponse)
async def create_job(req: BatchRequest):
    _validate(req)

    job_id = str(uuid.uuid4())
    now = time.time()
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "response": None,
        "error": None,
    }

    # Fire-and-forget background task (single-process demo).
    asyncio.create_task(_job_worker(job_id, req))

    return JobCreateResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
