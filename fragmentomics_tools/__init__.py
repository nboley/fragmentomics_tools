from __future__ import annotations

__all__ = [
    "FragmentArray",
    "RegionFragmentArray",
    "merge_fragment_arrays",
    "RegionDataFrame",
    "SampleAndRegionDataFrame",
    "SampleDataFrame",
    "Region",
]


def __getattr__(name: str):
    if name in ("FragmentArray", "RegionFragmentArray", "merge_fragment_arrays"):
        from .fragment_array import FragmentArray, RegionFragmentArray, merge_fragment_arrays
        globals().update(
            FragmentArray=FragmentArray,
            RegionFragmentArray=RegionFragmentArray,
            merge_fragment_arrays=merge_fragment_arrays,
        )
        return globals()[name]
    if name in ("RegionDataFrame", "SampleAndRegionDataFrame", "SampleDataFrame"):
        from .dataframe import RegionDataFrame, SampleAndRegionDataFrame, SampleDataFrame
        globals().update(
            RegionDataFrame=RegionDataFrame,
            SampleAndRegionDataFrame=SampleAndRegionDataFrame,
            SampleDataFrame=SampleDataFrame,
        )
        return globals()[name]
    if name == "Region":
        from .region import Region
        globals()["Region"] = Region
        return Region
    raise AttributeError(f"module 'fragmentomics_tools' has no attribute {name!r}")
