import json
import numpy as np
import zarr

STORE = "/efs/analytics/nathanboley/background_model/simulation_v3/stores/sim_store_v3_A.zarr"
SIM = "/efs/analytics/nathanboley/background_model/simulation_v3/A"

root = zarr.open_group(STORE, mode="r")
cfg = json.loads(root.attrs["config_json"])
print("config:", json.dumps(cfg, indent=1)[:2000])
print("config_hash", root.attrs["config_hash"], "split_version", root.attrs["split_version"])
print("arrays:", sorted(root.array_keys()), sorted(root.group_keys()))
for g in root.group_keys():
    for k in root[g].array_keys():
        a = root[f"{g}/{k}"]
        print(f"  {g}/{k} shape={a.shape} dtype={a.dtype}")

starts = root["tiles/start"][:]
stops = root["tiles/stop"][:]
splits = root["tiles/split"][:]
contigs = root["tiles/contig"][:]
roles = root["samples/role"][:]
print("n_tiles", len(starts), "span uniq", np.unique(stops - starts))
print("split counts", np.unique(splits, return_counts=True))
print("roles", np.unique(roles, return_counts=True))
print("first 3 tiles", [(str(contigs[i]), int(starts[i]), int(stops[i])) for i in range(3)])

mask = root["tiles/mask"]
print("mask shape", mask.shape, "all true?", bool(np.asarray(mask[:]).all()))

rt = np.load(SIM + "/region_table.npz", allow_pickle=True)
print("region_table keys", list(rt.keys()))
print("region_len", rt["region_len"])
print("rt n", len(rt["gstart"]), "span uniq", np.unique(rt["gstop"] - rt["gstart"]))
print("rt first3", [(str(rt["contig"][i]), int(rt["gstart"][i]), int(rt["gstop"][i])) for i in range(3)])

gt = np.load(SIM + "/ground_truth.npz", allow_pickle=True)
print("gt keys", list(gt.keys()))
for k in gt.keys():
    try:
        print("  ", k, np.shape(gt[k]), gt[k].dtype)
    except Exception as e:
        print("  ", k, "ERR", e)
print(open(SIM + "/ground_truth.json").read()[:3000])
