import numpy
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import nbinom

from fragmentomics_tools.fragment_array import merge_fragment_arrays


class UniformityFilter():
    @staticmethod
    def fit_alpha(sample):
        mu = sample.mean()
        nbfit = smf.negativebinomial("nbdata ~ 1", data=pd.DataFrame({"nbdata": sample})).fit_regularized(alpha=0, disp=False)
        return nbfit.params.iloc[1]

    def build_dist(self, mu=None):
        if mu is None:
            mu = self._mean
        mu = max(1./self._length, mu)
        var = mu + self._alpha * mu ** 2
        p = mu / var
        n = mu ** 2 / (var - mu)
        return nbinom(n=n, p=p)

    def find_ub(self, mu=None):
        return int(self.build_dist(mu=mu).ppf(1-1e-6)) # 1.0 - 1.0/self._length))

    def __init__(self, fa):
        cov = numpy.concatenate([fa.first_covered_base_counts, fa.last_covered_base_counts])
        self._region = fa.region
        self._length = cov.shape[0]
        self._n = cov.sum()
        self._mean = cov.mean()
        self._alpha = self.fit_alpha(cov)

    def __repr__(self):
        return f"<UniformityFilter alpha={self._alpha}>"

    def find_filter_indices(self, fa):
        ub = self.find_ub(float(fa.n_fragments)/self._length)
        return(
            numpy.nonzero(fa.first_covered_base_counts > ub)[0].ravel(),
            numpy.nonzero(fa.last_covered_base_counts > ub)[0].ravel()
        )

    @staticmethod
    def filter_fragment_array_from_indices(fa, first_filter_indices, last_filter_indices):
        frag_mask = numpy.zeros(fa.n_fragments, dtype=bool)
        for i in first_filter_indices:
            frag_mask |= (fa.first_covered_bases_0 == i)
        for i in last_filter_indices :
            frag_mask |= (fa.last_covered_bases_0 == i)
        return fa.mask(~frag_mask)

    def filter_fragment_array(self, fa):
        ub = self.find_ub(float(fa.n_fragments)/self._length)
        first_filter_indices, last_filter_indices = self.find_filter_indices(fa)
        return self.filter_fragment_array_from_indices(fa, first_filter_indices, last_filter_indices)


def apply_uniformity_filter_simple(srdf):
    pos_cols = ["contig", "start", "stop", "strand"]
    regions_grouped_srdf = srdf.df.groupby(pos_cols).progress_apply(lambda x: UniformityFilter(merge_fragment_arrays(x.fragment_array.tolist()))).rename("uniformity_filter")
    srdf = type(srdf)(srdf.df.set_index(pos_cols).join(regions_grouped_srdf).reset_index(), ref=srdf.ref)
    srdf['fragment_array'] = srdf.progress_apply(lambda x: x.uniformity_filter.filter_fragment_array(x.fragment_array), axis=1).rename("fragment_array")
    return srdf


def apply_uniformity_filter(srdf):
    srdf = srdf.copy()
    pos_cols = ["contig", "start", "stop", "strand"]
    merged_fas = srdf.df.groupby(pos_cols).progress_apply(lambda x: merge_fragment_arrays(x.fragment_array.tolist())).rename("fragment_array")
    filter_indices = merged_fas.progress_apply(lambda fa: UniformityFilter(fa).find_filter_indices(fa)).rename("filter_indices")
    tmp = srdf.df.set_index(pos_cols).join(filter_indices).reset_index()
    srdf['fragment_array'] = type(srdf)(tmp, ref=srdf.ref).parallel_apply(lambda x: UniformityFilter.filter_fragment_array_from_indices(x.fragment_array, x.filter_indices[0], x.filter_indices[1])).iloc[:, 0] # .rename("fragment_array")
    return srdf
