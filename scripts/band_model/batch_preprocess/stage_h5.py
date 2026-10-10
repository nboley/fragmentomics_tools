"""Stage the 50 draw h5s from EFS -> S3, keyed by library. Idempotent by size.

Only stdlib + boto3 (no numpy), so it runs on any host with EFS access and
AWS credentials.
"""

import argparse
import csv
import os

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, help="draw50.tsv (has efs_realpath col)")
    ap.add_argument("--s3-base", required=True, help="s3://bucket/prefix")
    ap.add_argument("--region", default="us-east-1")
    a = ap.parse_args()

    assert a.s3_base.startswith("s3://"), a.s3_base
    bucket, prefix = a.s3_base[len("s3://"):].split("/", 1)
    prefix = prefix.rstrip("/")
    s3 = boto3.client("s3", region_name=a.region)

    rows = list(csv.DictReader(open(a.draw), delimiter="\t"))
    for i, row in enumerate(rows, 1):
        lib = row["library"]
        src = row["efs_realpath"]
        key = f"{prefix}/inputs/h5/{lib}.h5"
        local_size = os.path.getsize(src)
        present = False
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            present = head["ContentLength"] == local_size
        except Exception:
            present = False
        if present:
            print(f"[{i}/{len(rows)}] skip {lib} (present {local_size/1e6:.0f} MB)")
            continue
        print(f"[{i}/{len(rows)}] upload {lib} ({local_size/1e6:.0f} MB)")
        s3.upload_file(src, bucket, key)
    print("h5 staging complete")


if __name__ == "__main__":
    main()
