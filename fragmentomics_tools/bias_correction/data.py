import pandas as pd
import numpy as np
import scipy

import torch
from tqdm import tqdm

tqdm.pandas()

from fragmentomics_tools.dataframe import SampleAndRegionDataFrame, RegionDataFrame
from fragmentomics_tools.fragment_array import merge_fragment_arrays

TILE_REGION_SIZE = 2048 * 8
FASTA_PATH = "/home/nboley/src/Ravel/data/repo_data_manifest/reference/GRCh38/GRCh38.p12.genome.fa.gz"

def _identity_fn(x):
    return x

def index_key_to_track_name(args):
    strand, fl, coverage_type = args
    return f"strand_{strand}__fl_{fl[0]}_{fl[1]}__coverage_{coverage_type}"

def track_name_to_index_key(track_name):
    return re.fullmatch("strand_({+-})__fl_(\d+)_(\d+)__coverage_(first|last|midpoint)", track_name)

def build_counts(record, return_sparse=False):
    compress = scipy.sparse.csr_array if return_sparse else _identity_fn        
        
    # drop duplicates
    rfa = record.fragment_array.drop_duplicate_fragments()
    if hasattr(record, 'blacklist_regions'):
        rfa = rfa.mask_overlapping_fragments(record.blacklist_regions, expansion=120)

    res = {}
    for strand in ['+', '-']:
        sub_rfa = rfa.subset_by_fragment_strand(strand)
        for fl in [(40, 65), (120, 175)]:
            sub_sub_rfa = sub_rfa.subset_fragment_lengths(*fl)
            key = (strand, fl, 'first')
            assert key not in res
            res[key] = compress(sub_sub_rfa.first_covered_base_counts)

            key = (strand, fl, 'last')
            assert key not in res
            res[key] = compress(sub_sub_rfa.last_covered_base_counts)

            key = (strand, fl, 'midpoint')
            assert key not in res
            res[key] = compress(sub_sub_rfa.get_midpoint_coverage_array())

    res = pd.Series(res)
    res.index = [index_key_to_track_name(x) for x in res.index]
    return res

def build_tss_coverage_counts(record):
    return build_counts(record)

def build_gene_coverage_counts(record):
    return build_counts(record)['strand_+__fl_40_65__coverage_first', 'strand_-__fl_40_65__coverage_last']

class FragmentEndpointsDataset(torch.utils.data.Dataset):
    @staticmethod
    def build_resized_one_hot_encoded_seq(rdf, input_region_size):
        rdf = RegionDataFrame(rdf[['contig', 'start', 'stop', 'strand']], ref=rdf.ref)
        sizes = list(rdf.region_lengths.drop_duplicates())
        assert len(sizes) == 1
        output_region_size = sizes[0]
        one_hot_encoded_sequences = rdf.resize_regions(
            input_region_size
        ).get_one_hot_encoded_sequence(FASTA_PATH)
        return one_hot_encoded_sequences

    def __init__(self, rdf, sdf, model, num_workers=24):
        self.rdf = rdf.copy()
        self.sdf = sdf.copy()
        self.output_columns = model.output_columns

        # init the srdf, and resize so that the regions are a multiple of the tile size, and then tile across the regions
        new_region_sizes = TILE_REGION_SIZE * (
            rdf.region_lengths / TILE_REGION_SIZE
        ).apply(np.ceil).astype(int)
        rdf = rdf.resize_regions(new_region_sizes)
        if "gene_id" in rdf.columns:
            rdf['id'] = rdf.gene_id
        else:
            rdf['id'] = [str(x) for x in rdf.iter_regions()]

        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(num_cores=num_workers, min_mapq=10)

        # merge fragment arrays
        fa_df = (
            pd.DataFrame(srdf)[["id", "fragment_array"]]
            .groupby("id")
            .progress_apply(
                lambda x: merge_fragment_arrays(
                    [x.drop_duplicate_fragments() for x in x.fragment_array]
                )
            )
        )
        fa_df.name = "fragment_array"
        srdf = rdf.set_index("id").join(fa_df).reset_index()
        srdf["sample_id"] = "merged"
        srdf["frag_h5"] = "none"
        srdf = SampleAndRegionDataFrame(srdf, ref=rdf.ref)

        if not all(x == TILE_REGION_SIZE for x in srdf.region_lengths.value_counts().index.tolist()):
            # TODO -- optimize by building sequence on rdf and then joining
            # note that this subsets the fragment arrays
            srdf = srdf.bin_regions_into_windows(TILE_REGION_SIZE, mode="exact")

        srdf["one_hot_encoded_sequence"] = self.build_resized_one_hot_encoded_seq(
            srdf, model.calc_input_region_size(TILE_REGION_SIZE)
        )
        srdf = srdf.attach_blacklist_regions(
            "/scratch/karius/annotation/mappability.simple_repeats.sorted.bed.gz"
        )
        self.srdf = srdf

    def __len__(self):
        return len(self.srdf)

    @staticmethod
    def get_targets_from_record(record, columns):
        valid_columns = [
            'strand_+__fl_40_65__coverage_first',
            'strand_+__fl_40_65__coverage_last',
            'strand_+__fl_40_65__coverage_midpoint',
            'strand_+__fl_120_175__coverage_first',
            'strand_+__fl_120_175__coverage_last',
            'strand_+__fl_120_175__coverage_midpoint',
            'strand_-__fl_40_65__coverage_first',
            'strand_-__fl_40_65__coverage_last',
            'strand_-__fl_40_65__coverage_midpoint',
            'strand_-__fl_120_175__coverage_first',
            'strand_-__fl_120_175__coverage_last',
            'strand_-__fl_120_175__coverage_midpoint'
        ]
        assert all(c in valid_columns for c in columns)

        targets = [torch.Tensor(x.astype(float))[:, None] for x in build_counts(record)[columns]]
        targets = torch.cat(targets, dim=1).T
        return targets

    def __getitem__(self, idx):
        record = self.srdf.iloc[idx, :]
        inputs = torch.Tensor(record.one_hot_encoded_sequence).cuda()
        targets = self.get_targets_from_record(record, self.output_columns).cuda()
        return inputs, targets
