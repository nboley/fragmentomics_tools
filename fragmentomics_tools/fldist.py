"""Fragment-length distribution."""

import numpy as np
import pandas as pd


class FlDist:
    @classmethod
    def init_from_sdf(cls, sdf):
        sample_ids = list(sdf["sample_id"])
        dupes = sorted(
            set(sid for sid in sample_ids if sample_ids.count(sid) > 1)
        )
        if dupes:
            raise ValueError(
                f"Duplicate sample_id(s) in SDF: {dupes}. "
                f"FlDist requires unique sample ids."
            )
        max_frag_len = 512
        columns = {}
        for record in sdf.itertuples():
            cnts = record.frag_h5.fragment_length_counts
            if len(cnts) < max_frag_len:
                cnts = np.pad(cnts, (0, max_frag_len - len(cnts)))
            else:
                cnts = cnts[:max_frag_len]
            total = cnts.sum()
            if total > 0:
                cnts = cnts / total
            columns[record.sample_id] = cnts

        fl_df = pd.DataFrame(columns)
        fl_df = fl_df.set_index(fl_df.index + 1)

        return cls(fl_df)

    def subset_by_sample_ids(self, sample_ids):
        missing = set(sample_ids) - set(self.fl_df.columns)
        if missing:
            raise KeyError(
                f"sample_id(s) not found in FlDist: {sorted(missing)}"
            )
        fl_df = self.fl_df.T.loc[sample_ids].T
        return type(self)(fl_df)

    def __init__(self, fl_df):
        self.fl_df = fl_df

    def plot(
        self,
        figsize=(20, 8),
        legend=False,
        max_frag_len=None,
        include_reference=True,
    ):
        import seaborn as sns

        sns.set(rc={"figure.figsize": figsize})

        fl_df = self.fl_df.copy()
        if include_reference:
            ref_fl_dist = fl_df.mean(axis=1)
            ref_fl_dist = ref_fl_dist / ref_fl_dist.sum()
            fl_df.loc[:, "Reference"] = ref_fl_dist

        return fl_df.loc[0:max_frag_len, :].plot(legend=legend)
