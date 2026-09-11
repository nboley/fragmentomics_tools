"""AWS Batch orchestrator for the background-model store preprocess.

Runs on a host with AWS creds + EFS access (this laptop/host, NOT Batch).
Sequence:
  1. register a container job definition (biomarker image, 2 vCPU / 8 GB),
  2. submit the 1-sample PILOT (plain job, INDEX=0),
  3. wait, read its timing.json from S3, evaluate the GATE
     (wall <= ~30 min = within 2x of the 5-15 min/sample design band),
  4. on PASS -> AUTO-CONTINUE: submit the 50-task array (resume skips index 0)
     then the single Phase-B job,
  5. sync the finished store from S3 back to EFS,
  6. deregister the job definition.

Code reaches the tasks via `git clone` of a PUBLIC branch/sha (bootstrap in the
job-def command). The exact SHA is pinned. Nothing is baked into the image.

Requires: boto3 + the background_model.config import (no numpy) for the hash.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time

import boto3

# background_model.config imports only stdlib -> safe to import here.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from background_model.config import PlumbingConfig  # noqa: E402

# ── Defaults (verified 2026-09-11) ───────────────────────────────────────
REGION = "us-east-1"
IMAGE = "573640641260.dkr.ecr.us-east-1.amazonaws.com/omni/biomarker:0.2.2"
QUEUE = "nextflow-pipeline-b-ecs"  # spot fleet compute env
REPO_URL = "https://github.com/nboley/fragmentomics_tools.git"
BRANCH = "background-model-v2"
# Resolved from the local git HEAD at runtime (see resolve_sha) so the tasks
# clone exactly the code being orchestrated — avoids a self-referential pin.
SHA = None
S3_BASE = "s3://fragmentomics.kariusdx.com/nboley/bg-preprocess"
JOBDEF_NAME = "bg-preprocess"
VCPUS = "2"
MEMORY = "8192"
REF = "hg38"
N_SAMPLES = 50
GATE_WALL_SEC = 30 * 60  # 2x the upper end of the 5-15 min/sample design band
EFS_STORE_DIR = "/efs/analytics/nathanboley/background_model/stores"

# Bootstrap: clone the pinned code, then exec the requested entrypoint.
BOOTSTRAP = r"""
set -euo pipefail
export WORKDIR="${WORKDIR:-/tmp/bgwork}"
mkdir -p "$WORKDIR"
command -v git >/dev/null 2>&1 || (microdnf install -y git 2>/dev/null || yum install -y git 2>/dev/null || (apt-get update && apt-get install -y git) 2>/dev/null) || true
echo "[bootstrap] clone $REPO_URL@$BRANCH ($SHA)"
git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR/repo"
git -C "$WORKDIR/repo" checkout --quiet "$SHA"
export REPO_DIR="$WORKDIR/repo"
exec bash "$REPO_DIR/scripts/batch_preprocess/$ENTRYPOINT"
""".strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _s3_split(uri):
    assert uri.startswith("s3://")
    bucket, key = uri[len("s3://"):].split("/", 1)
    return bucket, key


def compute_hash8_and_lib0():
    """Deterministic store hash8 + first-library, from the local input files."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg = PlumbingConfig(
        sample_sheet=os.path.join(repo, "data/sample_sheets/ibd_quiescent.tsv"),
        region_beds={"train_pool": os.path.join(repo, "data/region_sets/training_tiles.bed")},
        blacklist_bed=os.path.join(repo, "data/region_sets/exclusion_4_blacklist_encode_v2.bed"),
        fasta="/efs/analytics/nathanboley/data_resources/genome/hg38.fa",
    )
    import csv
    with open(os.path.join(repo, "data/sample_sheets/draw50.tsv")) as f:
        lib0 = next(csv.DictReader(f, delimiter="\t"))["library"]
    return cfg.config_hash8(), cfg.store_name(), lib0


def resolve_sha():
    if SHA:
        return SHA
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()


def base_environment():
    return [
        {"name": "REPO_URL", "value": REPO_URL},
        {"name": "BRANCH", "value": BRANCH},
        {"name": "SHA", "value": resolve_sha()},
        {"name": "S3_BASE", "value": S3_BASE},
        {"name": "REF", "value": REF},
        {"name": "AWS_DEFAULT_REGION", "value": REGION},
    ]


def register_jobdef(batch):
    resp = batch.register_job_definition(
        jobDefinitionName=JOBDEF_NAME,
        type="container",
        containerProperties={
            "image": IMAGE,
            "command": ["bash", "-c", BOOTSTRAP],
            "resourceRequirements": [
                {"type": "VCPU", "value": VCPUS},
                {"type": "MEMORY", "value": MEMORY},
            ],
            "environment": base_environment(),
        },
    )
    jd = f"{resp['jobDefinitionName']}:{resp['revision']}"
    log(f"registered job definition {jd}")
    return jd


def submit_job(batch, jd, name, entrypoint, index=None, array_size=None):
    env = [{"name": "ENTRYPOINT", "value": entrypoint}]
    if index is not None:
        env.append({"name": "INDEX", "value": str(index)})
    kwargs = dict(
        jobName=name,
        jobQueue=QUEUE,
        jobDefinition=jd,
        containerOverrides={"environment": env},
    )
    if array_size is not None:
        kwargs["arrayProperties"] = {"size": array_size}
    resp = batch.submit_job(**kwargs)
    log(f"submitted {name} -> {resp['jobId']}")
    return resp["jobId"]


def wait_single(batch, job_id, poll=30, max_wait=6 * 3600):
    t0 = time.time()
    last = None
    while time.time() - t0 < max_wait:
        j = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        st = j["status"]
        if st != last:
            log(f"  {job_id} -> {st}")
            last = st
        if st == "SUCCEEDED":
            return True, j
        if st == "FAILED":
            return False, j
        time.sleep(poll)
    return False, {"status": "TIMEOUT"}


def wait_array(batch, job_id, poll=60, max_wait=12 * 3600):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        j = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        summ = j.get("arrayProperties", {}).get("statusSummary", {}) or {}
        done = summ.get("SUCCEEDED", 0)
        failed = summ.get("FAILED", 0)
        total = j.get("arrayProperties", {}).get("size", 0)
        log(f"  array {job_id}: {summ}")
        if total and done + failed >= total:
            return failed == 0, j
        time.sleep(poll)
    return False, {"status": "TIMEOUT"}


def read_timing(s3, hash8, lib0):
    bucket, base = _s3_split(S3_BASE)
    key = f"{base}/shards_{hash8}/{lib0}.timing.json"
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return json.loads(buf.getvalue().decode())


def evaluate_gate(timing):
    wall = timing["wall_sec"]
    band = "within 5-15 min band" if 300 <= wall <= 900 else (
        "faster than band" if wall < 300 else "slower than band"
    )
    passed = wall <= GATE_WALL_SEC
    log(f"GATE: wall={wall:.0f}s ({wall/60:.1f} min); {band}; "
        f"threshold={GATE_WALL_SEC}s -> {'PASS' if passed else 'FAIL'}")
    return passed


def efs_sync_back(hash8, store_name):
    dst = os.path.join(EFS_STORE_DIR, store_name)
    os.makedirs(EFS_STORE_DIR, exist_ok=True)
    src = f"{S3_BASE}/store/{store_name}/"
    log(f"EFS sync-back: {src} -> {dst}")
    subprocess.run(["aws", "s3", "cp", "--recursive", src, dst,
                    "--region", REGION], check=True)
    subprocess.run(["aws", "s3", "cp",
                    f"{S3_BASE}/store/bg_store_{hash8}.config.json",
                    os.path.join(EFS_STORE_DIR, f"bg_store_{hash8}.config.json"),
                    "--region", REGION], check=True)
    log(f"EFS store ready: {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-only", action="store_true",
                    help="submit + gate the pilot, then stop (no auto-continue)")
    ap.add_argument("--skip-pilot", action="store_true",
                    help="skip pilot/gate and go straight to array + Phase B")
    ap.add_argument("--no-efs-sync", action="store_true")
    a = ap.parse_args()

    hash8, store_name, lib0 = compute_hash8_and_lib0()
    log(f"store={store_name} hash8={hash8} pilot-library={lib0}")
    log(f"pinned code: {REPO_URL}@{BRANCH} sha={resolve_sha()}")

    batch = boto3.client("batch", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    jd = register_jobdef(batch)

    try:
        if not a.skip_pilot:
            pid = submit_job(batch, jd, "bg-preprocess-pilot", "task_entrypoint.sh", index=0)
            ok, j = wait_single(batch, pid)
            if not ok:
                log(f"PILOT FAILED: {j.get('statusReason')} "
                    f"attempts={j.get('attempts')}")
                return 2
            timing = read_timing(s3, hash8, lib0)
            log(f"PILOT timing: {json.dumps(timing)}")
            if not evaluate_gate(timing):
                log("Gate FAILED -> stopping (no auto-continue). Report metrics.")
                return 3
            if a.pilot_only:
                log("--pilot-only: stopping after gate PASS.")
                return 0

        # AUTO-CONTINUE: full 50-task array (resume skips index 0), then Phase B.
        aid = submit_job(batch, jd, "bg-preprocess-array", "task_entrypoint.sh",
                         array_size=N_SAMPLES)
        ok, j = wait_array(batch, aid)
        if not ok:
            log(f"ARRAY had failures: {j.get('arrayProperties', {}).get('statusSummary')}")
            return 4

        bid = submit_job(batch, jd, "bg-preprocess-phaseb", "phase_b_entrypoint.sh")
        ok, j = wait_single(batch, bid)
        if not ok:
            log(f"PHASE B FAILED: {j.get('statusReason')}")
            return 5

        if not a.no_efs_sync:
            efs_sync_back(hash8, store_name)
        log(f"COMPLETE: store={store_name} S3={S3_BASE}/store/{store_name}/")
        return 0
    finally:
        try:
            batch.deregister_job_definition(jobDefinition=jd)
            log(f"deregistered {jd}")
        except Exception as e:  # noqa: BLE001
            log(f"deregister failed (non-fatal): {e}")


if __name__ == "__main__":
    raise SystemExit(main())
