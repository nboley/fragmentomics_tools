import warnings
from typing import List, Dict, Union

import numpy
import pandas
from joblib import Parallel, delayed
from pyranges import pyranges
from tqdm import tqdm

from fbio.dataframe import RegionDataFrame
from ravel.learn.transforms import indicator_vect_to_one_hot_mat


class ChromhmmData:
    """
    Reads and handles ChromHMM data. Input must be a bed file with 4 columns: ["Chromosome", "Start", "End", "State"],
    where "State" either starts with E, e.g., E15, or is named -- e.g., 15_quies

    Multiple ChromHMM annotations can be passed concurrently, and ChromhmmData will return annotations for all of them.
    The standard usage involves reading in the ChromHMM data and passing regions through get_annotations_from_regions.
    This returns a

    ChromHMM annotations can also be mixed by providing fractions
    """

    def _get_format(self):
        """
        :return: ChromHMM annotations have two different formats -- the ones that start with "E", and "named"
        """

        def _get_format_for_df(_chromhmm_df):
            _first_state = _chromhmm_df.iloc[0].State
            if _first_state.startswith("E"):
                return "E"  # E15
            elif "_" in _first_state:
                try:
                    int(_first_state.split("_")[0])
                except ValueError:
                    raise NotImplementedError(f"{_first_state} not in recognized format")
                return "named"  # 15_quies
            else:
                raise NotImplementedError(f"{_first_state} not in recognized format")

        first_state = _get_format_for_df(self.chromhmm_dfs[0])
        for cc, chromhmm_df in enumerate(self.chromhmm_dfs[1:]):
            assert first_state == _get_format_for_df(chromhmm_df), (
                f"ChromHMM BEDs {self.chromhmm_bed_paths[0]} and {self.chromhmm_bed_paths[cc+1]} "
                f"have incompatible formats"
            )
        return first_state

    def _get_index(self, state):
        """
        :param state: the named annotation directly from ChromHMM annotation lines
        :return: The index the annotation corresponds to
        """
        if self.chromhmm_format == "E":
            return int(state[1:]) - 1
        elif self.chromhmm_format == "named":
            return int(state.split("_")[0]) - 1
        else:
            raise NotImplementedError(f"{self.chromhmm_format} not implemented")

    def _get_index_to_names(self):
        """
        :return: a dict mapping index (0 through num_states-1) to the named state
        """
        index_to_name = {}
        for name in set(self.chromhmm_dfs[0].State):
            state_index = self._get_index(name)
            index_to_name[state_index] = name
        return index_to_name

    def _get_most_common_state(self, _chromhmm_df):
        _state_counts = numpy.zeros(self.num_chromhmm_states)
        for gg, group in _chromhmm_df.groupby("State"):
            _state_counts[self._get_index(gg)] = (group["End"] - group["Start"]).sum()
        return numpy.argmax(_state_counts)

    def _infer_quiescent_state(self):
        """
        Because the beginning of chromosome 1 is typically marked quiescent (due to low mappability), we can infer
        that the first state observed is quiescent
        :return:
        """
        first_state = self._get_most_common_state(self.chromhmm_dfs[0])
        for cc, chromhmm_df in enumerate(self.chromhmm_dfs[1:]):
            this_quiescent_state = self._get_most_common_state(chromhmm_df)
            assert first_state == this_quiescent_state, (
                f"ChromHMM bed {self.chromhmm_bed_paths[cc+1]} has a different inferred quiescent state "
                f"from {self.chromhmm_bed_paths[0]}: {this_quiescent_state} vs {first_state}"
            )
        return first_state

    def _infer_num_states(self):
        """
        :return: the number of states observed. Checks if all files have the same number of states
        """

        def _get_num_obs_states(_chromhmm_df):
            return len(set(_chromhmm_df.State))

        num_obs_states = _get_num_obs_states(self.chromhmm_dfs[0])
        for cc, chromhmm_df in enumerate(self.chromhmm_dfs[1:]):
            this_num_obs_states = _get_num_obs_states(chromhmm_df)
            assert num_obs_states == this_num_obs_states, (
                f"{self.chromhmm_bed_paths[cc+1]} had {this_num_obs_states}"
                f"instead of an expected {num_obs_states}"
            )
        return num_obs_states

    def read_chromhmm_dataframes(self):
        """Reads in the ChromHMM bed into a dataframe"""
        """
        # file 1
        chr1    0   10000   15
        chr1    10000 10600   12
        ...
        # file 2
        chr1    0   10000   15
        chr1    10000 10600 11
        chr1    10600 19000 12
        ...
        """
        return [
            pandas.read_csv(
                chromhmm_bed_path,
                sep="\t",
                names=["Chromosome", "Start", "End", "State"],
                nrows=1000 if self.test else None,
            )
            for chromhmm_bed_path in self.chromhmm_bed_paths
        ]

    def _get_chromhmm_starts_ends_states(self, chromhmm_df):
        """
        This uses pyranges to find intersections between pairs of genomic intervals. It's super fast!
        https://pyranges.readthedocs.io/en/master/autoapi/pyranges/index.html
        :param chromhmm_df: a ChromHMM dataframe with columns Chromosome, Start, End, State
        :return: Super compressed version of ChromHMM annotations. For the given ChromHMM DataFrame, it returns a
         dictionary with four keys: starts, ends, states, and region_length
        """

        def _start_pos(row):
            return max(row["Start"] - row["Start_region"], 0)

        def _end_pos(row):
            _region_size = row["End_region"] - row["Start_region"]
            return min(row["End"] - row["Start_region"], _region_size)

        chromhmm_pr = pyranges.PyRanges(chromhmm_df)
        regions_pr = pyranges.PyRanges(
            pandas.DataFrame(
                self.region_dataframe.rename(
                    {"contig": "Chromosome", "start": "Start", "stop": "End"}, axis=1
                )
            )
        )
        joined_pr = chromhmm_pr.join(regions_pr, suffix="_region")
        num_regions = len(self.region_dataframe)
        # We use a dict because we need quick per-region lookup. The key is the index of the region dataframe
        starts = {}
        ends = {}
        states = {}
        region_size = {}

        num_regions_with_incomplete_chromhmm = 0

        for gg, group in joined_pr.df.groupby("peak_index"):
            # starts is the beginning position in the region of each ChromHMM state (starting at 0)
            starts[gg] = []
            # ends is the ending position of the state
            ends[gg] = []
            # states is the state (in index) format
            states[gg] = []
            # region_size is the length of the region in the dataframe
            region_size[gg] = group.iloc[0].End_region - group.iloc[0].Start_region
            for state, start, end in zip(
                group.State,
                group.apply(_start_pos, axis=1),
                group.apply(_end_pos, axis=1),
            ):
                assert end >= start, f"Cannot have negative length interval in {group}: {start, end}"
                state_idx = self._get_index(state)
                starts[gg].append(start)
                ends[gg].append(end)
                states[gg].append(state_idx)

            # In np.piecewise, if there is an extra func (state), then that is used as the default value
            # This takes the last position in states -- if some position is missing in the piecewise function,
            # it defaults to this state
            states[gg].append(self.fill_state_idx)
            if starts[gg][0] != 0 or ends[gg][-1] != region_size[gg]:
                # All regions that do not overlap with a ChromHMM annotation get marked as fill_state_idx
                # Typically, you want this to be the quiescent or low state
                if self.allow_incomplete_chromhmm:
                    num_regions_with_incomplete_chromhmm += 1
                else:
                    raise ValueError(
                        f"{gg}, {group} does not have length {region_size[gg]}. "
                        f"This may be due to an incomplete ChromHMM annotation. "
                        f"Try allow_incomplete_chromhmm=True"
                    )

        num_regions_with_no_chromhmm = 0
        for ii in range(num_regions):
            if ii not in starts:
                expected_length = (
                    self.region_dataframe.loc[ii]["stop"] - self.region_dataframe.loc[ii]["start"]
                )
                starts[ii] = [0]
                ends[ii] = [expected_length]
                states[ii] = [self.fill_state_idx]
                region_size[ii] = expected_length
                num_regions_with_no_chromhmm += 1

        if self.allow_incomplete_chromhmm:
            if num_regions_with_incomplete_chromhmm > 0:
                warnings.warn(
                    f"Found {num_regions_with_incomplete_chromhmm}/{num_regions} regions "
                    f"with ChromHMM covering fewer than expected number of bases"
                )
            if num_regions_with_no_chromhmm > 0:
                warnings.warn(
                    f"Found {num_regions_with_no_chromhmm}/{num_regions} regions with no overlapping ChromHMM"
                )
        else:
            assert num_regions_with_incomplete_chromhmm == num_regions_with_no_chromhmm == 0

        return {"starts": starts, "ends": ends, "states": states, "region_size": region_size}

    def _cache_chromhmm_starts_ends_states_parallel(self, disable_tqdm=False, workers=1):
        data = Parallel(workers)(
            delayed(self._get_chromhmm_starts_ends_states)(chromhmm_df)
            for chromhmm_df in tqdm(
                self.chromhmm_dfs,
                desc="Caching ChromHMM annotations",
                total=len(self.chromhmm_dfs),
                disable=disable_tqdm,
            )
        )

        return data

    def _cache_chromhmm_starts_ends_states(self, disable_tqdm=False):
        # Keeping this separate for debugging purposes
        data = [
            self._get_chromhmm_starts_ends_states(chromhmm_df)
            for chromhmm_df in tqdm(
                self.chromhmm_dfs,
                desc="Caching ChromHMM annotations",
                total=len(self.chromhmm_dfs),
                disable=disable_tqdm,
            )
        ]

        return data

    def cache_chromhmm_starts_ends_states(self, disable_tqdm=False, workers=1):
        if workers == 1:
            return self._cache_chromhmm_starts_ends_states(disable_tqdm=disable_tqdm)
        else:
            return self._cache_chromhmm_starts_ends_states_parallel(
                disable_tqdm=disable_tqdm, workers=workers
            )

    def __init__(
        self,
        chromhmm_bed_paths: List[str],
        region_dataframe: RegionDataFrame,
        fill_state_idx: Union[int, None] = None,
        num_states: Union[int, None] = None,
        ref: str = "hg38",
        output_format: str = "one_hot",
        allow_incomplete_chromhmm: bool = False,
        show_progress: bool = True,
        test: bool = False,
        workers: int = 1,
    ):
        self.test = test  # This will cause only reading in only the first 20 lines

        if isinstance(chromhmm_bed_paths, str):
            chromhmm_bed_paths = [chromhmm_bed_paths]
        self.chromhmm_bed_paths = chromhmm_bed_paths
        self.region_dataframe = region_dataframe
        self.region_dataframe["peak_index"] = self.region_dataframe.index
        self.ref = ref
        assert self.ref == self.region_dataframe.ref
        self.chromhmm_dfs = self.read_chromhmm_dataframes()

        self.chromhmm_format = self._get_format()
        self.num_chromhmm_states = self._infer_num_states() if num_states is None else num_states
        self.index_to_names = self._get_index_to_names()
        self.output_format = output_format

        self.fill_state_idx = self._infer_quiescent_state() if fill_state_idx is None else fill_state_idx
        self.allow_incomplete_chromhmm = allow_incomplete_chromhmm

        # This is the big step -- intersects the ChromHMM dataframes with the region dataframe and makes it usable
        self.data: List[Dict[str, Dict[str, List]]] = self.cache_chromhmm_starts_ends_states(
            disable_tqdm=not show_progress,
            workers=workers,
        )

    @staticmethod
    def get_chromhmm_indicator_vect(starts, ends, states, region_size):
        """
        Uses numpy's piecewise to go from starts, ends, and states to a vector of values, then makes into a bool
        :param starts: a list of starting positions of a ChromHMM annotations in region
        :param ends: a list of ending positions of a ChromHMM annotations in region
        :param states: a list of ChromHMM states corresponding to [(start, end) for start, stop in zip(starts, ends)]
        :param region_size:
        :return: a boolean matrix indicating ChromHMM states throughout the desired region (region_length, num_states)
        """
        positions = numpy.arange(region_size)
        indicator_vect = numpy.piecewise(
            positions,
            [(start <= positions) & (positions < end) for start, end in zip(starts, ends)],
            states,
        )

        return indicator_vect

    def get_indexes_for_region_idx(self, idx):
        index_vects = [
            self.get_chromhmm_indicator_vect(
                _data["starts"][idx], _data["ends"][idx], _data["states"][idx], _data["region_size"][idx]
            )
            for _data in self.data
        ]
        return index_vects

    def get_names_for_region_idx(self, idx):
        index_vects = self.get_indexes_for_region_idx(idx)
        return [[self.index_to_names[index] for index in index_vect] for index_vect in index_vects]

    def get_one_hot_for_region_idx(self, idx):
        index_vects = self.get_indexes_for_region_idx(idx)
        return [
            indicator_vect_to_one_hot_mat(index_vect, self.num_chromhmm_states) for index_vect in index_vects
        ]

    def get_data_for_region_idx(self, idx: int, output_format: str):
        """
        Returns a matrix of size (num_celltypes, num_positions, num_chromhmm_states)
        """
        if output_format == "index":
            output = self.get_indexes_for_region_idx(idx)
        elif output_format == "names":
            output = self.get_names_for_region_idx(idx)
        elif output_format == "one_hot":
            output = self.get_one_hot_for_region_idx(idx)
        else:
            raise NotImplementedError(f"output format {output_format} is not implemented")
        return numpy.array(output)

    def set_output_format(self, output_format):
        valid_output_formats = ["index", "names", "one_hot"]
        assert output_format in valid_output_formats, f"output_format must be one of {valid_output_formats}"
        self.output_format = output_format

    def __len__(self):
        return len(self.region_dataframe)

    def __getitem__(self, idx):
        assert (
            self.output_format is not None
        ), "output_format is None. Set with ChromhmmData.set_output_format"
        return self.get_data_for_region_idx(idx, self.output_format)
