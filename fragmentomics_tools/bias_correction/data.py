import pandas as pd
import numpy as np

import torch
from tqdm import tqdm

tqdm.pandas()

from fragmentomics_tools.dataframe import SampleAndRegionDataFrame, RegionDataFrame
from fragmentomics_tools.fragment_array import merge_fragment_arrays

TILE_REGION_SIZE = 2048 * 8
FASTA_PATH = "/home/nboley/src/Ravel/data/repo_data_manifest/reference/GRCh38/GRCh38.p12.genome.fa.gz"


class FragmentEndpointsDataset(torch.utils.data.Dataset):
    @staticmethod
    def build_resized_one_hot_encoded_seq(rdf, input_region_size):
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

        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(num_cores=num_workers, min_mapq=10)

        # merge fragment arrays
        fa_df = (
            pd.DataFrame(srdf)[["gene_id", "fragment_array"]]
            .groupby("gene_id")
            .progress_apply(
                lambda x: merge_fragment_arrays(
                    [x.drop_duplicate_fragments() for x in x.fragment_array]
                )
            )
        )
        fa_df.name = "fragment_array"
        srdf = rdf.set_index("gene_id").join(fa_df).reset_index()
        srdf["sample_id"] = "merged"
        srdf["frag_h5"] = "none"
        srdf = SampleAndRegionDataFrame(srdf, ref=rdf.ref)

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
    def get_targets_from_region_fragment_array(rfa, columns, blacklist_regions):
        valid_columns = [
            "fwd_start_counts",
            "bkwd_start_counts",
            "fwd_stop_counts",
            "bkwd_stop_counts",
        ]
        assert all(column in valid_columns for column in columns)
        rfa = rfa.mask_overlapping_fragments(blacklist_regions, expansion=120)
        rfa = rfa.subset_fragment_lengths(40, 60)
        fwd_fa = rfa.subset_by_fragment_strand("+")
        bkwd_fa = rfa.subset_by_fragment_strand("-")

        def get_cov(column):
            if column == "fwd_start_counts":
                return torch.Tensor(fwd_fa.first_covered_base_counts.astype(float))[
                    :, None
                ]
            if column == "fwd_stop_counts":
                return torch.Tensor(fwd_fa.last_covered_base_counts.astype(float))[
                    :, None
                ]
            if column == "bkwd_start_counts":
                return torch.Tensor(bkwd_fa.first_covered_base_counts.astype(float))[
                    :, None
                ]
            if column == "bkwd_stop_counts":
                return torch.Tensor(bkwd_fa.last_covered_base_counts.astype(float))[
                    :, None
                ]

        targets = [get_cov(column) for column in columns]
        targets = torch.cat(targets, dim=1).T
        return targets

    def __getitem__(self, idx):
        record = self.srdf.iloc[idx, :]
        inputs = torch.Tensor(record.one_hot_encoded_sequence).cuda()
        targets = self.get_targets_from_region_fragment_array(
            record.fragment_array, self.output_columns, record.blacklist_regions
        ).cuda()
        return inputs, targets
