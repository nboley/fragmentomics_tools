"""Fast checkpoint hooks for fragmentomics_tools custom types.

These hooks are registered with the dill checkpointer at kernel startup.
Discovery happens via the ``claude_mcp.checkpoint_hooks`` entry point declared
in pyproject.toml.

Each hook pair consists of:
  save_fn(obj)    -> state   (must be efficiently dill-serializable)
  restore_fn(state) -> obj

The state for array-heavy types uses plain numpy arrays and bytes so that
dill's numpy protocol (raw buffer copy) is used rather than generic object
serialization — this is orders of magnitude faster for large arrays.
"""

import io


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

def _region_to_dict(region):
    return {
        "chrom": region.chrom,
        "start": region.start,
        "stop": region.stop,
        "strand": region.strand,
        "ref": region.ref,
        "data": region.data,
    }


def _region_from_dict(d):
    from fragmentomics_tools.region import Region
    return Region(
        chrom=d["chrom"],
        start=d["start"],
        stop=d["stop"],
        strand=d.get("strand"),
        ref=d.get("ref"),
        data=d.get("data"),
    )


# ---------------------------------------------------------------------------
# FragmentArray / RegionFragmentArray
# ---------------------------------------------------------------------------

def _save_fragment_array(fa):
    """Serialize a FragmentArray as a dict of numpy arrays + scalars.

    All numpy arrays are stored directly so dill uses its fast numpy protocol
    (raw buffer copy) instead of generic Python object pickling.
    """
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray

    state = {
        "starts_0": fa.starts_0,
        "stops_0": fa.stops_0,
        "length": fa.length,
        "max_frag_len": fa.max_frag_len,
        "weights": fa.weights,
        "first_covered_base_weights": fa.first_covered_base_weights,
        "last_covered_base_weights": fa.last_covered_base_weights,
        "fragment_strands": fa.fragment_strands,  # numpy array or None
        "num_cpgs": fa.num_cpgs,
        "num_converted_cpgs": fa.num_converted_cpgs,
        "num_cytosines": fa.num_cytosines,
        "num_converted_cytosines": fa.num_converted_cytosines,
        "is_flipped": fa.is_flipped,
        "_is_region": isinstance(fa, RegionFragmentArray),
    }
    if isinstance(fa, RegionFragmentArray):
        state["region"] = _region_to_dict(fa.region)
    return state


def _restore_fragment_array(state):
    from fragmentomics_tools.fragment_array.fragment_array import (
        FragmentArray,
        RegionFragmentArray,
    )

    common = dict(
        starts_0=state["starts_0"],
        stops_0=state["stops_0"],
        max_frag_len=state["max_frag_len"],
        validate_data=False,
        fragment_strands=state["fragment_strands"],
        weights=state["weights"],
        first_covered_base_weights=state["first_covered_base_weights"],
        last_covered_base_weights=state["last_covered_base_weights"],
        num_cpgs=state["num_cpgs"],
        num_converted_cpgs=state["num_converted_cpgs"],
        num_cytosines=state["num_cytosines"],
        num_converted_cytosines=state["num_converted_cytosines"],
        is_flipped=state["is_flipped"],
    )

    if state.get("_is_region"):
        return RegionFragmentArray(region=_region_from_dict(state["region"]), **common)
    else:
        return FragmentArray(length=state["length"], **common)


# ---------------------------------------------------------------------------
# RegionDataFrame / SampleAndRegionDataFrame
# ---------------------------------------------------------------------------

def _save_region_dataframe(rdf):
    """Serialize a RegionDataFrame as parquet bytes + metadata.

    Preserves the ``ref`` metadata attribute and any FragmentArray columns
    that would otherwise block parquet serialization.

    Fast path: if the whole DataFrame serializes to parquet cleanly, do that
    and just save ref alongside it.  Only when parquet fails do we extract
    the specific blocking columns.
    """
    from fragmentomics_tools.fragment_array.fragment_array import FragmentArray

    cls_name = type(rdf).__name__
    ref = rdf.ref
    col_order = list(rdf.columns)

    # Fast path: try the full DataFrame directly.
    buf = io.BytesIO()
    try:
        rdf.to_parquet(buf)
        return {
            "cls": cls_name,
            "ref": ref,
            "parquet_bytes": buf.getvalue(),
            "obj_cols": {},
            "col_order": col_order,
        }
    except Exception:
        pass

    # Slow path: parquet failed (e.g. a column holds Python objects like
    # FragmentArrays).  Find the blocking columns, extract them, and retry
    # parquet on the remainder.
    import pandas as pd

    plain_df = pd.DataFrame(rdf)
    obj_cols: dict = {}
    drop_cols: list = []

    for col in plain_df.columns:
        if plain_df[col].dtype != object:
            continue
        # Test whether this individual column blocks parquet.
        try:
            plain_df[[col]].to_parquet(io.BytesIO())
            continue  # parquet-compatible — leave it in
        except Exception:
            pass
        # Column blocks parquet — extract it.
        first = next((x for x in plain_df[col] if x is not None), None)
        if first is not None and isinstance(first, FragmentArray):
            obj_cols[col] = {
                "_type": "fragment_array_list",
                "data": [
                    _save_fragment_array(fa) if fa is not None else None
                    for fa in plain_df[col]
                ],
            }
        else:
            import dill
            obj_cols[col] = {
                "_type": "dill_bytes",
                "data": dill.dumps(plain_df[col].tolist()),
            }
        drop_cols.append(col)

    serializable_df = plain_df.drop(columns=drop_cols) if drop_cols else plain_df
    buf = io.BytesIO()
    serializable_df.to_parquet(buf)

    return {
        "cls": cls_name,
        "ref": ref,
        "parquet_bytes": buf.getvalue(),
        "obj_cols": obj_cols,
        "col_order": col_order,
    }


def _restore_region_dataframe(state):
    import pandas as pd
    from fragmentomics_tools.dataframe import RegionDataFrame, SampleAndRegionDataFrame

    df = pd.read_parquet(io.BytesIO(state["parquet_bytes"]))

    for col, col_info in state.get("obj_cols", {}).items():
        if col_info["_type"] == "fragment_array_list":
            df[col] = [
                _restore_fragment_array(s) if s is not None else None
                for s in col_info["data"]
            ]
        elif col_info["_type"] == "dill_bytes":
            import dill
            df[col] = dill.loads(col_info["data"])

    # Restore original column order.
    col_order = [c for c in state.get("col_order", df.columns) if c in df.columns]
    df = df[col_order]

    cls_map = {
        "RegionDataFrame": RegionDataFrame,
        "SampleAndRegionDataFrame": SampleAndRegionDataFrame,
    }
    cls = cls_map.get(state["cls"], RegionDataFrame)
    return cls(df, ref=state["ref"])


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------

def register_checkpoint_hooks(checkpointer):
    """Register fragmentomics_tools type hooks with a DillCheckpointer.

    Called automatically at kernel startup via the ``claude_mcp.checkpoint_hooks``
    entry point, or manually:

        from fragmentomics_tools.checkpoint_hooks import register_checkpoint_hooks
        register_checkpoint_hooks(checkpointer)
    """
    from fragmentomics_tools.fragment_array.fragment_array import (
        FragmentArray,
        RegionFragmentArray,
    )
    from fragmentomics_tools.dataframe import RegionDataFrame, SampleAndRegionDataFrame

    # Register most-specific types first so MRO lookup in the checkpointer
    # matches the right hook when subclasses are involved.
    checkpointer.register(RegionFragmentArray, _save_fragment_array, _restore_fragment_array)
    checkpointer.register(FragmentArray, _save_fragment_array, _restore_fragment_array)
    checkpointer.register(SampleAndRegionDataFrame, _save_region_dataframe, _restore_region_dataframe)
    checkpointer.register(RegionDataFrame, _save_region_dataframe, _restore_region_dataframe)
