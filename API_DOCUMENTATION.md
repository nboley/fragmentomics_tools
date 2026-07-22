# fragmentomics_tools API Documentation

## Table of Contents

- [Region & Fragment](#region-&-fragment)
- [Core DataFrames](#core-dataframes)
- [Fragment Arrays](#fragment-arrays)
- [Plotting](#plotting)

---

## Region & Fragment

### region.py

#### `OutOfBoundsError`

**Location**: region.py:30

#### `ChromOrdering`

**Location**: region.py:34

#### `Region`

**Location**: region.py:65

**Methods**:

- `__contains__(self, other)` (region.py:696)
  - Tests whether one region is a subregion of another
- `__eq__(self, other)` (region.py:676)
  - >>> Region('chr1', 1, 2) == Region('chr1', 1, 2)
- `__gt__(self, other)` (region.py:652)
  - >>> Region('chr1', 200, 300) > Region('chr1', 100, 200)
- `__hash__(self)` (region.py:717)
- `__lt__(self, other)` (region.py:642)
  - >>> Region('chr1', 100, 200) < Region('chr1', 200, 300)
- `__ne__(self, other)` (region.py:660)
  - >>> Region('chr1', 100, 200) != Region('chr1', 100, 201)
- `__post_init__(self)` (region.py:503)
  - >>> Region('chr2', 1, 2)
- `__repr__(self)` (region.py:667)
- `__str__(self)` (region.py:670)
- `bed_strand(self)` (region.py:734)
  - Return strand if strand in '+-.' and '.' if None
- `cmp(self, region, chrom_ordering)` (region.py:602)
  - If regions intersect, return 0
- `convert_region(self, unique_liftoverer: RegionLiftOver)` (region.py:317)
  - Returns a copy of converted into the new reference.
- `five_prime_resize(self, new_length)` (region.py:119)
- `five_prime_shift(self, offset: int)` (region.py:100)
  - Shift the five prime end of region 'offset' basepairs towards the three prime end.
- `flip_strand(self)` (region.py:77)
- `from_region_str(cls, region_str, ref: str, data)` (region.py:497)
- `get_annotation_coverage_array(self, annotation_names)` (region.py:247)
  - :param annotation_names: ex: ['repeat_masker'] one of fbio.annotation.ANNOTATION.keys()
- `get_bed_coverage_array(self, bed_path)` (region.py:287)
  - :param bed_path:
- `get_coverage_array(self, regions)` (region.py:213)
  - Gets a coverage array of the number of pileup intersections with the regions in regions.
- `get_one_hot_encoded_sequence(self, reference_fasta, reverse_complement_sequence_if_minus_strand: bool)` (region.py:803)
  - get the sequence of a region from a fasta file
- `get_overlaps_repeat_or_blacklist_mask_array(self)` (region.py:260)
  - :return: A boolean array which is True everywhere the region intersects a repeat element or an element
- `get_resize_start(start: int, current_size: int, new_size: int, strand)` (region.py:357)
- `get_resize_starts(start, current_size, new_size, strand)` (region.py:373)
- `get_sequence(self, reference_fasta)` (region.py:793)
  - get the sequence of a region from a fasta file
- `get_subregion_from_jitter_and_strand(self, output_size: int, jitter: int, strand)` (region.py:822)
- `intersect(self, other: Region)` (region.py:144)
  - >>> x = Region('chr1', 10, 20)
- `intersect_annotation(self, annotation_name: str)` (region.py:185)
  - Intersects this Region with a annotation_name
- `intersect_with_bed(self, bed_path)` (region.py:275)
- `intersects(self, other: Region)` (region.py:588)
  - >>> x = Region('chr1', 10, 20)
- `is_minus_strand(self)` (region.py:789)
  - Return True if the strand is set and it is -
- `is_subregion(self, other: Region)` (region.py:568)
  - Returns True if other is contained within self.
- `left_resize(self, new_length)` (region.py:86)
- `left_shift(self, offset: int)` (region.py:83)
- `length(self)` (region.py:565)
- `liftover(self)` (region.py:348)
  - Alias for convert_region.
- `midpoint(self)` (region.py:353)
- `name(self)` (region.py:74)
- `parse_region_str(s)` (region.py:460)
  - Parses a string into chrom, start, stop, strand
- `plot_annotations(self, annotation_names)` (region.py:294)
  - Plots annotation tracks over this region
- `random(cls, length, assembly, chroms, strand)` (region.py:746)
  - :param length: length of the region
- `resize(self, new_size, truncate_when_out_of_bounds)` (region.py:399)
  - You can think about resizing as an iterative process
- `right_resize(self, new_length)` (region.py:92)
- `right_shift(self, offset: int)` (region.py:89)
- `shift(self, n)` (region.py:769)
  - Shifts a region by n bases.  Use a negative to shift upstream, positive to shift it downstream
- `strand_is_set(self)` (region.py:782)
  - Return true if the strand is set to something, also verify it is valid.
- `three_prime_resize(self, new_length)` (region.py:141)
- `three_prime_shift(self, offset: int)` (region.py:122)
  - Shift the three prime end of region 'offset' basepairs in the three prime direction.
- `to_bed3_line(self)` (region.py:729)
  - contig start stop
- `to_bed6_line(self, name, score)` (region.py:741)
  - contig start stop
- `truncate(left_amt, right_amt)` (region.py:95)

### fragment.py

#### `ClassPropertyDescriptor`

**Location**: fragment.py:6

**Description**: used by classproperty() to make @classproperty decorator

**Methods**:

- `__get__(self, obj, klass)` (fragment.py:15)
- `__init__(self, fget, fset)` (fragment.py:11)
- `__set__(self, obj, value)` (fragment.py:20)
- `setter(self, func)` (fragment.py:26)

#### `Fragment`

**Location**: fragment.py:44

**Methods**:

- `__eq__(self, other)` (fragment.py:210)
- `__init__(self, chrom: str, start: int, stop: int, mapq1, mapq2, gc, strand, cell_barcode, num_cpgs, num_meth_cpgs)` (fragment.py:58)
- `__repr__(self)` (fragment.py:202)
- `field_names(cls)` (fragment.py:126)
- `field_types(cls)` (fragment.py:130)
- `from_bed12_parts(cls, parts)` (fragment.py:243)
- `from_line(cls, line)` (fragment.py:187)
- `from_line_parts(cls, parts)` (fragment.py:150)
- `length(self)` (fragment.py:195)
- `length_and_midpoint_to_start_and_stop(length, midpoint)` (fragment.py:216)
  - Converts length and midpoint representation to start and stop representation.  Note that there is a
- `mapq12_min(self)` (fragment.py:94)
  - Returns the minimum of self.mapq1 and self.mapq2 ignoring None values
- `mapq_gte(self, threshold)` (fragment.py:80)
  - >>> Fragment('chr1', 0, 1, mapq1=9).mapq_gte(10)
- `midpoint(self)` (fragment.py:199)
- `replace(self)` (fragment.py:120)
- `tlen(self)` (fragment.py:191)
- `to_line(self)` (fragment.py:134)

### contig.py

#### `ReferenceInferenceError`

**Location**: contig.py:45

#### `ChromOrdering`

**Location**: contig.py:212

## Core DataFrames

### dataframe.py

#### `DataFrameBase`

**Location**: dataframe.py:152

**Methods**:

- `__init__(self, data)` (dataframe.py:197)
- `_constructor(self)` (dataframe.py:161)
- `_parallel_apply(self, fn, n_workers, verbose)` (dataframe.py:240)
- `_repr_html_(self)` (dataframe.py:169)
- `df(self)` (dataframe.py:193)
  - Cast to a normal pandas dataframe
- `from_fname_s3_or_local(cls, fname)` (dataframe.py:183)
- `parallel_apply(self, fn, n_workers, verbose)` (dataframe.py:300)
- `reorder_columns(self)` (dataframe.py:179)
- `to_string(self)` (dataframe.py:172)
  - When formatting for pandas dfs columns are sometimes dropped and then the

#### `RegionDataFrame`

**Location**: dataframe.py:332

**Methods**:

- `__and__(self, other)` (dataframe.py:383)
- `__eq__(self, other)` (dataframe.py:401)
  - Checks if two region dataframes have identical regions, in the same order
- `__init__(self, data)` (dataframe.py:362)
- `_error_on_invalid_new_starts(new_start)` (dataframe.py:1314)
- `_error_on_invalid_new_stops(rdf, new_stop)` (dataframe.py:1322)
- `_get_fragment_coverage_sum(self, in_fname: str, sorted)` (dataframe.py:1240)
  - For each region, get the fragment coverage for in_fname, which is a fragment coverage bigwig or fragment bed
- `_get_fragment_coverage_track(self, in_fname: str)` (dataframe.py:1193)
  - :param in_fname: a bigwig or fragments h5 file
- `_get_seq(self, fasta_path, seq_type, reverse_complement_sequence_if_minus_strand, verbose)` (dataframe.py:1620)
- `_required_columns(self)` (dataframe.py:346)
- `_resize_region_boundaries(self, left: int, right: int, inplace: bool, strand_aware: bool, discard_invalid_resizes: bool)` (dataframe.py:1352)
- `_valid_regions_mask(self, new_start, new_stop, discard_buffer_bp)` (dataframe.py:1340)
- `annotate_regions_with_max_tf_scores(self, target_tfs, target_len: int, num_workers: int, batch_size: int, cuda: bool, inplace: bool, default_jaspar: bool)` (dataframe.py:756)
  - Add TF related columns and set start/stop to be the start/stop of the tf. Each region in the output will be `target_len` long. The max tf score involves
- `attach_blacklist_regions(self, bed_fname)` (dataframe.py:1147)
- `attach_num_tss_overlaps(self, tss_intervals, from_midpoint: bool, expand_upstream: int, expand_downstream: int, conservative: bool)` (dataframe.py:551)
  - :param tss_intervals: If provided, will skip re-querying the TSS intervals
- `attach_one_hot_encoded_sequence(self)` (dataframe.py:1684)
- `attach_sequence(self)` (dataframe.py:1666)
- `bases_overlap_with_bed(self, bed_file)` (dataframe.py:1115)
  - Finds intersection between RegionDataFrame and some bed file, returning the number of overlapping bases
- `bases_overlap_with_beds(self, bed_files, num_cores)` (dataframe.py:1123)
  - Finds intersection between RDF and a list of bed files, returning list of number of overlapping bases
- `bed_df(self)` (dataframe.py:1168)
- `bin_regions_into_windows(self, window_size, mode, stride)` (dataframe.py:1466)
  - Multiply all regions by tiling windows across each region in self.
- `center_on_summit(self)` (dataframe.py:608)
  - Center regions on summit, resize the regions, and then drop the summit column.
- `center_regions_on_tf_motif(self, target_tfs, target_len: int, tf_search_width, num_workers: int, batch_size: int, cuda: bool, verify_motif_scores: bool, inplace: bool, unique_regions: bool, shuffle_tf_motif: bool, default_jaspar: bool, shuffle_tf_seed: int, quiet: bool)` (dataframe.py:627)
  - Add TF related columns and set start/stop to be the start/stop of the tf. Each region in the output will be
- `concat(rdfs)` (dataframe.py:393)
- `downsample_stratified_by_label(self, max_features_per_label)` (dataframe.py:1839)
  - Downsample so that there is at most `max_number_of_samples_per_label` for each label.  Useful for quick
- `drop_overlapping_regions(self, other_rdf)` (dataframe.py:1136)
  - Masks blacklist regions, returning a new dataframe with regions that don't overlap blacklist regions.
- `expand_regions(left_amt: int, right_amt: int, inplace: bool, strand_aware: bool, discard_invalid_resizes: bool)` (dataframe.py:1391)
- `from_bed(cls, in_bed_file, ref)` (dataframe.py:417)
  - Convenience function to load from a bed file.
- `from_beds_merged(cls, in_bed_files, ref, chroms, bed_filter_callback)` (dataframe.py:432)
  - Reads in multiple beds, concatenates and merges them, safely truncating to get rid of strand
- `from_fname_s3_or_local(cls, fname)` (dataframe.py:356)
- `from_random_regions(cls, size, region_lengths, ref, chroms)` (dataframe.py:922)
  - Create a RegionDataFrame from a random set of regions
- `from_regions(cls, regions, ref: str)` (dataframe.py:897)
  - Create a RegionDataFrame from a set of regions
- `get_fasta_path(self)` (dataframe.py:349)
- `get_fragment_coverage_sum(self, in_fnames, num_cores, sorted, verbose)` (dataframe.py:1296)
  - For each region, return the sum of the number of reads in in_fnames
- `get_fragment_coverage_track(self, in_fnames, num_cores, verbose)` (dataframe.py:1223)
  - :param in_fnames: file or list of bigwig or fragment h5 files
- `get_interval_dict(self, data_cols, expand_upstream: int, expand_downstream: int)` (dataframe.py:492)
  - :param data_cols: if None, use dataframe's row index as the return
- `get_one_hot_encoded_sequence(self, fasta_path, reverse_complement_sequence_if_minus_strand, verbose)` (dataframe.py:1671)
- `get_overlapping_base_counts(self, bed_file, rsuff, sorted)` (dataframe.py:1056)
  - Returning the number of overlapping bases in an intersection between a RDF and bed file
- `get_pfm(self, reverse_complement_sequence_if_minus_strand, verbose)` (dataframe.py:1689)
  - Get the pfm by stacking up the sequence over all regions.
- `get_pwm(self)` (dataframe.py:1726)
- `get_sequence(self, fasta_path, reverse_complement_sequence_if_minus_strand, verbose)` (dataframe.py:1653)
- `intersect_with_bed(self, bed_file_path, sorted, rsuff)` (dataframe.py:991)
  - Finds intersection between RegionDataFrame and some bed file
- `intersect_with_rdf(self, other, sorted, rsuff)` (dataframe.py:947)
  - Creates the intersection of RegionDataFrames
- `iter_region_row(self)` (dataframe.py:1591)
  - iterates over dataframe rows with the region, each item yielded will be: (region, row)
- `iter_regions(self)` (dataframe.py:1179)
- `label_balanced(self, column_name, random_state)` (dataframe.py:1750)
  - Return a copy of self with balanced labels.
- `lift_over(self, new_ref, transfer_columns, remove_non_liftoverable_regions)` (dataframe.py:1018)
  - Lifts over RegionDataFrame to a new reference
- `merge_regions(self)` (dataframe.py:940)
- `nrow(self)` (dataframe.py:413)
- `overlaps_rdf(self, query: RegionDataFrame, max_distance: int)` (dataframe.py:525)
  - Returns a boolean series of which regions overlap the other dataframe
- `overlaps_with_bed(self, bed_file, invert, min_size)` (dataframe.py:1084)
  - Finds intersection between RegionDataFrame and some bed file, returning a True/False array
- `overlaps_with_beds(self, bed_files, num_cores)` (dataframe.py:1100)
  - Finds intersection between RegionDataFrame and a list of bed files, returning a list of True/False arrays
- `rdf_from_bed3(cls, fname, ref, nrows, label)` (dataframe.py:887)
- `ref_path(self)` (dataframe.py:587)
- `region_lengths(self)` (dataframe.py:489)
- `region_mask(self, region)` (dataframe.py:1160)
  - Returns a mask of this dataframe of all regions which intersect region
- `resize_regions(self, new_size, inplace: bool, discard_invalid_resizes: bool, discard_buffer_bp: int)` (dataframe.py:1421)
- `save_as_bed(self, path)` (dataframe.py:1176)
- `set_binary_label(self, on_query, off_query, drop_unlabeled_records, inplace, label_column)` (dataframe.py:1761)
  - Add a label column.
- `set_binary_label_by_thresholds(self, columns, on_threshold, off_threshold, drop_unlabeled_records, inplace, label_column)` (dataframe.py:1801)
  - Applies on/off thresholds to columns and sets the label column.
- `sort(self, inplace)` (dataframe.py:1008)
  - Sorts dataframe by contig, start, and stop (same as pybedtools)
- `split(self, num_sections)` (dataframe.py:1729)
- `split_on_column(self, column_name, value_groups)` (dataframe.py:1545)
- `split_on_contig(self, contig_groups)` (dataframe.py:1571)
- `split_on_query(self, query)` (dataframe.py:1534)
  - Split self into two dataframes.
- `to_tsv(self, path_or_buf, columns)` (dataframe.py:1598)
  - Using this function to write regions dataframes to disk facilitates writing and also
- `truncate_regions(left_amt: int, right_amt: int, inplace: bool, strand_aware: bool, discard_invalid_resizes: bool)` (dataframe.py:1406)
- `unique_regions(self, by, best_by, ascending_best: bool)` (dataframe.py:856)
  - Remove

#### `SampleAndRegionDataFrame`

**Location**: dataframe.py:1930

**Methods**:

- `_check_has_fragment_array(self)` (dataframe.py:1950)
- `_resize_region_boundaries(self, left: int, right: int, inplace: bool, strand_aware: bool, discard_invalid_resizes: bool)` (dataframe.py:2027)
- `attach_fragment_arrays(self)` (dataframe.py:1998)
- `bin_regions_into_windows(self)` (dataframe.py:2007)
- `expand_regions(self)` (dataframe.py:2054)
- `filter_outlier_counts(self, min_frags, num_sd, return_stat_columns)` (dataframe.py:2094)
- `get_sample_count_bounds(self, num_sd)` (dataframe.py:2079)
- `has_fragment_array(self)` (dataframe.py:1947)
- `init_from_rdf_and_sdf(cls, rdf, sdf)` (dataframe.py:1943)
- `load_fragment_arrays(self, n_workers, verbose, max_frag_len: int, generate_weights_callback, fragment_array_callback)` (dataframe.py:1956)
- `reorder_columns(self)` (dataframe.py:1933)
- `reset_fragment_array_weights(self)` (dataframe.py:2155)
  - Set the fragment array weights to zero.
- `resize_regions(self, new_size)` (dataframe.py:2062)
- `set_fragment_array_gc_weights(self, normalizer)` (dataframe.py:2139)
- `set_fragment_array_weights(self, model, n_workers)` (dataframe.py:2119)

#### `FlDist`

**Location**: dataframe.py:2163

**Methods**:

- `__init__(self, fl_df)` (dataframe.py:2183)
- `init_from_sdf(cls, sdf)` (dataframe.py:2165)
- `plot(self, figsize, legend, max_frag_len, include_reference)` (dataframe.py:2187)
- `subset_by_sample_ids(self, sample_ids)` (dataframe.py:2179)

#### `SampleDataFrame`

**Location**: dataframe.py:2200

**Methods**:

- `__init__(self, data)` (dataframe.py:2204)
- `dropna(self)` (dataframe.py:2216)
- `fl_dist(self)` (dataframe.py:2220)
- `label_balanced(self, column_name, random_state)` (dataframe.py:2223)
  - Return a copy of self with balanced labels.

## Fragment Arrays

### fragment_array.py

#### `SparseIntVector`

**Location**: fragment_array.py:53

**Methods**:

- `__add__(self, other)` (fragment_array.py:64)
- `__init__(self, coords, data, length, check)` (fragment_array.py:54)
- `__radd__(self, other)` (fragment_array.py:70)
- `__repr__(self)` (fragment_array.py:79)
- `__str__(self)` (fragment_array.py:82)
- `sum(ars)` (fragment_array.py:86)
- `todense(self)` (fragment_array.py:76)

#### `FragmentDoesNotIntersect`

**Location**: fragment_array.py:150

#### `InvalidCoordinates`

**Location**: fragment_array.py:154

#### `FragmentArray`

**Location**: fragment_array.py:158

**Description**: A FragmentArray is a collection of fragments stored in a spare-array-like coordinate format.  The
fragment coordinates (starts_0 and stops_0) are the coordinates relative to the region start and region end.

It is distinct from a FragmentMatrix because a FragmentMatrix stores midpoints/lengths, and has the limitation of
filtering out fragments which intersect a region, but who's midpoints are out of bounds.

>>> fa = FragmentArray(starts_0=[-1,2,3], stops_0=[3,4,5], length=5, max_frag_len=10)
>>> fa.starts_0
array([-1,  2,  3], dtype=int32)
>>> fa.stops_0
array([3, 4, 5], dtype=int32)

Counts of number of fragment starts over this region. Out of bounds starts are not counted.
>>> fa.first_covered_base_counts
array([0., 0., 1., 1., 0.])

Counts of the number of fragment ends over this region (ends are stops-1).  Out of bounds ends are not counted.
Note that this is generally more useful than fa.stop_counts, since the "end" is the specific position the fragment
ends.
>>> fa.last_covered_base_counts
array([0., 0., 1., 1., 1.])

You can add two FragmentArrays from identical regions (example, same region over two different BAMs)
>>> fa2 = FragmentArray(starts_0=[3,4,4], stops_0=[6,7,8], length=5, max_frag_len=10)
>>> fa + fa2
FragmentArray(n_frags=6, length=5, starts_0=[-1, 2, 3, 3, 4, 4], stops_0=[3, 4, 5, 6, 7, 8], weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], first_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], last_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], num_cpgs=[0, 0, 0, 0, 0, 0], num_meth_cpgs=[0, 0, 0, 0, 0, 0], max_frag_len=10)

You can add two FragmentArrays over different regions as long as the region lengths are the same.  It
produces a fragment array with a "Pseudo-Region" (a region with "NA" for its chromosome, and a start of 0)
>>> fa2 = FragmentArray(starts_0=[-1,2,3], stops_0=[3,4,5], length=5, max_frag_len=10)
>>> fa + fa2
FragmentArray(n_frags=6, length=5, starts_0=[-1, 2, 3, -1, 2, 3], stops_0=[3, 4, 5, 3, 4, 5], weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], first_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], last_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], num_cpgs=[0, 0, 0, 0, 0, 0], num_meth_cpgs=[0, 0, 0, 0, 0, 0], max_frag_len=10)

**Methods**:

- `__add__(self, other)` (fragment_array.py:398)
  - Adds two FragmentArrays together.  Both FragmentArrays must have the same region length.
- `__eq__(self, other: FragmentArray)` (fragment_array.py:472)
  - >>> fa1 = FragmentArray([-1,2,3], [3,4,500], 100, 511)
- `__init__(self, starts_0, stops_0, length: int, max_frag_len: int, validate_data: bool, fragment_strands, weights, first_covered_base_weights, last_covered_base_weights, num_cpgs, num_converted_cpgs, num_cytosines, num_converted_cytosines, is_flipped: bool)` (fragment_array.py:213)
  - A FragmentArray is a collection of fragments stored in a spare-array-like coordinate format.  The
- `__radd__(self, other)` (fragment_array.py:384)
- `__repr__(self)` (fragment_array.py:755)
- `__str__(self)` (fragment_array.py:738)
- `_get_covered_base_array(self, positions_attr, weights_attr, return_sparse)` (fragment_array.py:803)
- `_ones_if_none(self, _x)` (fragment_array.py:201)
- `_replace(self, validate_data: bool)` (fragment_array.py:374)
  - Reinitalize self, potentially replacing input args with entries from **kwargs
- `_resize(self, new_size: int)` (fragment_array.py:631)
- `_shift_boundaries(left, right, validate_data)` (fragment_array.py:603)
  - Modify the length of self.region without changin fragments.
- `_subset_or_mask(self, mask, validate_data)` (fragment_array.py:936)
  - Keep fragments that satisfy mask.
- `_zeros_if_none(self, _x)` (fragment_array.py:207)
- `arr(self)` (fragment_array.py:923)
- `build_coverage_counts(self, fl_bands, split_strand, return_sparse)` (fragment_array.py:889)
- `dense_array(self)` (fragment_array.py:731)
  - Returns a dense array who's axes are fragment_length/midpoint over the region
- `downsampled(self, n, random_state)` (fragment_array.py:537)
  - :param n: number of fragments to keep
- `downsampled_frag_lens(self, frag_len_acceptance_prbs)` (fragment_array.py:560)
  - Downsample self where the acceptance probability for a fragment is taken from frag_len_acceptance_prbs.
- `drop_duplicate_fragments(self)` (fragment_array.py:992)
- `filter_by_methyl(self, num_cpgs, num_meth_cpgs, pct_meth_cpgs)` (fragment_array.py:1042)
  - Filters by the min and max number of CpGs, max and min number that are methylated, and / or pct methylated
- `first_covered_base_counts(self)` (fragment_array.py:838)
- `first_covered_bases_0(self)` (fragment_array.py:783)
- `frag_str(frags)` (fragment_array.py:361)
  - Dump out a pretty formatted string for a fragment position array.
- `fragment_lengths(self)` (fragment_array.py:771)
- `fragment_matrix(self)` (fragment_array.py:932)
- `from_frag_length_midpoint_dense_array(cls, dense_array)` (fragment_array.py:1082)
  - Converts a dense array of frag_length/midpoint to start/stop sparse array coordinates,
- `get_first_covered_base_array(self, return_sparse)` (fragment_array.py:827)
  - 1d array of fragment start counts at each position that intersect self.region
- `get_fragment_coverage_array(self)` (fragment_array.py:865)
  - Get a vector of fragment pileup coverage
- `get_last_covered_base_array(self, return_sparse)` (fragment_array.py:841)
  - 1d array of fragment end counts at each position that intersect the region
- `get_midpoint_coverage_array(self, return_sparse)` (fragment_array.py:858)
- `jitter(self, jitter_value: int, output_length: int)` (fragment_array.py:700)
- `last_covered_base_counts(self)` (fragment_array.py:855)
- `last_covered_bases_0(self)` (fragment_array.py:779)
- `left_resize(self, new_length)` (fragment_array.py:678)
  - Resize self to new_length by modifying self.start
- `lengths(self)` (fragment_array.py:775)
- `make_tracks(self, vplot_sum_pool_by, vplot_label: str, vplot_cmap: str, binding_site_data, coverage_smoothing_window, show_coverage: bool)` (fragment_array.py:1120)
  - Creates a Tracks instance for plotting this FragmentMatrix
- `mask(self, mask, validate_data)` (fragment_array.py:960)
- `midpoint_0(self)` (fragment_array.py:791)
- `midpoint_covered_base_counts(self)` (fragment_array.py:862)
- `midpoints_0(self)` (fragment_array.py:787)
- `n_fragments(self)` (fragment_array.py:763)
- `n_frags(self)` (fragment_array.py:759)
- `new_starts_stops_mask_for_resize(self, new_size: int)` (fragment_array.py:624)
- `oversampled(self, n)` (fragment_array.py:574)
  - Takes fragments, returns all of them, plus some oversampled
- `pct_meth_cpgs(self)` (fragment_array.py:352)
- `plot(self)` (fragment_array.py:1231)
- `plot_region(self)` (fragment_array.py:195)
  - The Region instance to use for plotting
- `reset_cutsite_bias_weights(self)` (fragment_array.py:369)
- `resize(self, new_size: int)` (fragment_array.py:639)
- `resize_offset(self, new_size: int, region)` (fragment_array.py:612)
  - Return the integer offset for fragment starts/ends when asking for
- `reverse_strand(self)` (fragment_array.py:703)
  - Create a copy of self with the strand and fragment matrix reversed
- `right_resize(self, new_length)` (fragment_array.py:689)
  - Resize self to new_length by modifying self.stop
- `sample_with_replacement(self, n)` (fragment_array.py:600)
- `sampled_to_frac(self, frac, random_state)` (fragment_array.py:592)
- `shape(self)` (fragment_array.py:767)
- `shift_and_zero_pad(self, shift_amt)` (fragment_array.py:642)
  - Shift all fragments by shift_amt and then remove all fragments that don't overlap self.
- `sort_in_place(self)` (fragment_array.py:446)
- `split_into_2_nonoverlapping_fas(self, sample_size, seed)` (fragment_array.py:1028)
  - Split into 2 fragment arrays with 'sample_size' distinct fragments in the first, and the
- `split_into_k_nonoverlapping_fas(self, sample_size, k)` (fragment_array.py:1019)
  - Split into 'k' fragment arrays with 'sample_size' distinct fragments in each.
- `split_into_nonoverlapping_fas(self, sample_sizes, seed)` (fragment_array.py:998)
  - Split into fragment arrays with 'sample_size' distinct fragments in each.
- `subset(self, indices, validate_data)` (fragment_array.py:963)
  - Return a subset of self for the fragmnets in indices.
- `subset_by_fragment_strand(self, strand)` (fragment_array.py:970)
  - Return two fragment arrays each containing the fragments on either strand.
- `subset_fragment_lengths(self, min_frag_len, max_frag_len)` (fragment_array.py:794)
- `to_fragment_matrix(self)` (fragment_array.py:918)
- `truncate(left_amt, right_amt)` (fragment_array.py:661)
  - Make self smaller by left_amt and/or right_amt
- `valid_idxs(starts_0, stops_0, length: int)` (fragment_array.py:514)
  - Returns a boolean array to filter out fragments that do not overlap a desired length
- `vplot(self, sum_pool_by, title)` (fragment_array.py:1071)
- `zero_pad(self, left_amt, right_amt)` (fragment_array.py:651)

#### `RegionFragmentArray`

**Location**: fragment_array.py:1235

**Methods**:

- `__add__(self, other)` (fragment_array.py:1506)
  - Adds two FragmentArrays together.  Both FragmentArrays must have the same region length.
- `__eq__(self, other: RegionFragmentArray)` (fragment_array.py:1582)
  - >>> fa1 = RegionFragmentArray([-1,2,3], [3,4,500], Region('chr1', 0, 100), 511)
- `__init__(self, starts_0, stops_0, region: Region, max_frag_len: int, validate_data: bool, fragment_strands, weights, first_covered_base_weights, last_covered_base_weights, num_cpgs, num_converted_cpgs, num_cytosines, num_converted_cytosines, is_flipped: bool)` (fragment_array.py:1251)
- `__str__(self)` (fragment_array.py:1404)
- `_shift_boundaries(left, right, validate_data)` (fragment_array.py:1334)
  - Modify the length of self.region without changin fragments.
- `chrom(self)` (fragment_array.py:1484)
- `five_prime_resize(self, new_length)` (fragment_array.py:1347)
  - Resize self by modifying the five prime end.
- `fragment_matrix(self)` (fragment_array.py:1479)
- `from_fname(cls, fname: str, region: Region, min_mapq: int, max_frag_len: int, include_fragment_strand, flip_data_to_match_region_strand, background_model, min_background_scaling_factor: float, max_background_scaling_factor: float)` (fragment_array.py:1817)
- `from_frag_bed(cls, in_frag_bed: str, region: Region, min_mapq: int, max_frag_len: int)` (fragment_array.py:1622)
  - Read an indexed frag bed.
- `from_frag_length_midpoint_dense_array(cls, dense_array, region)` (fragment_array.py:1471)
- `from_fragments_h5(cls, in_fragments_h5, region: Region, max_frag_len: int, generate_weights_callback)` (fragment_array.py:1706)
  - :param flip_data_to_match_region_strand: If True, the data is placed on the strand matching the region strand (if set).
- `init_from_fragment_array(cls, fa: FragmentArray, region: Region)` (fragment_array.py:1246)
  - Convenience function to add a region toa  base fragmnet array
- `is_minus_strand(self)` (fragment_array.py:1499)
- `load(cls, fname)` (fragment_array.py:1439)
- `make_data_direction_match_strand(self)` (fragment_array.py:1307)
- `mask_overlapping_fragments(self, mask_regions, expansion)` (fragment_array.py:1390)
- `midpoint(self)` (fragment_array.py:1503)
- `midpoint_0(self)` (fragment_array.py:1455)
- `midpoints(self)` (fragment_array.py:1467)
- `plot_region(self)` (fragment_array.py:1237)
  - The Region instance to use for plotting
- `resize(self, new_size: int)` (fragment_array.py:1320)
- `resize_offset(self, new_size: int)` (fragment_array.py:1343)
  - Return the offset for fragment starts/ends when asking for a resize of this region
- `reverse_strand(self)` (fragment_array.py:1299)
  - Create a copy of self with the strand and fragment matrix reversed
- `save(self, fname)` (fragment_array.py:1422)
- `shift_and_zero_pad(self, shift_amt)` (fragment_array.py:1326)
- `start(self)` (fragment_array.py:1492)
- `starts(self)` (fragment_array.py:1459)
- `stop(self)` (fragment_array.py:1496)
- `stops(self)` (fragment_array.py:1463)
- `strand(self)` (fragment_array.py:1488)
- `subset_by_region(self, subregion)` (fragment_array.py:1375)
- `three_prime_resize(self, new_length)` (fragment_array.py:1359)
  - Resize self by modifying the three prime end.

### fragment_matrix.py

#### `FragmentMatrix`

**Location**: fragment_matrix.py:58

**Methods**:

- `__add__(self, other)` (fragment_matrix.py:71)
- `__post_init__(self)` (fragment_matrix.py:64)
- `__radd__(self, other)` (fragment_matrix.py:95)
- `dense_array(self)` (fragment_matrix.py:103)
- `from_fragment_matrices(fragment_matrices)` (fragment_matrix.py:99)
- `get_coverage_density(self, pseudo_count)` (fragment_matrix.py:117)
- `get_fragment_length_density(self, pseudo_count)` (fragment_matrix.py:113)
- `reverse_sum_pooled(self, sum_pool_by, preserve_sum: bool)` (fragment_matrix.py:125)
  - Upscale the array.
- `smooth_by_sum_pool(self, sum_pool_by, preserve_sum: bool)` (fragment_matrix.py:146)
- `sum_pooled(self, sum_pool_by)` (fragment_matrix.py:121)
- `todense(self)` (fragment_matrix.py:106)
  - Return a fragment matrix array.

#### `RegionFragmentMatrix`

**Location**: fragment_matrix.py:153

**Methods**:

- `__post_init__(self)` (fragment_matrix.py:196)
- `chrom(self)` (fragment_matrix.py:177)
- `fragment_matrix(self)` (fragment_matrix.py:201)
- `get_slice(self, sl)` (fragment_matrix.py:208)
- `get_subregion_slice(self, subregion)` (fragment_matrix.py:232)
- `midpoint(self)` (fragment_matrix.py:193)
- `plot_region(self)` (fragment_matrix.py:158)
  - The Region to use for plotting.  It's overridden, since the FragmentMatrix plots over a pseudo-region (a
- `region_str(self)` (fragment_matrix.py:205)
- `start(self)` (fragment_matrix.py:185)
- `starts(self)` (fragment_matrix.py:166)
- `stop(self)` (fragment_matrix.py:189)
- `stops(self)` (fragment_matrix.py:170)
- `strand(self)` (fragment_matrix.py:181)

#### `TooFewReadsToDownsample`

**Location**: fragment_matrix.py:288

## Plotting

### tracks.py

#### `VLine`

**Location**: tracks.py:242

**Description**: Vertical Line plot object (at an x coordinate)

#### `Line`

**Location**: tracks.py:253

**Description**: Line plot object (y values over a region)

#### `GenomeTrack`

**Location**: tracks.py:265

**Description**: Base class for a Genome Track

**Methods**:

- `__add__(self, other)` (tracks.py:287)
- `__post_init__(self)` (tracks.py:346)
- `__radd__(self, other)` (tracks.py:293)
- `_materialize_data(self)` (tracks.py:364)
- `_plot(self, ax)` (tracks.py:355)
- `_plot_extras(self, ax)` (tracks.py:309)
- `_plot_left(self, ax_main, ax_right)` (tracks.py:361)
- `_plot_right(self, ax_main, ax_right)` (tracks.py:358)
- `chrom(self)` (tracks.py:319)
- `data(self)` (tracks.py:333)
  - The cached result of self._materialize_data()
- `label(self)` (tracks.py:305)
- `plot(self, ax_main, plot_region, ax_right)` (tracks.py:367)
  - Create a plot of all tracks
- `replace(self)` (tracks.py:315)
- `start(self)` (tracks.py:323)
- `stop(self)` (tracks.py:327)
- `title(self)` (tracks.py:301)

#### `Tracks`

**Location**: tracks.py:442

**Methods**:

- `get_igv_link(self, local_mount, region, ipython_link)` (tracks.py:447)
  - Load all tracks into IGV.  Prepends 'local_mout' to track.data
- `plot(self, plot_region, width, height_multiplier, vlines, out_fname, title, plot_region_coords, sharey)` (tracks.py:461)
  - :param plot_region: Override the regions being plotted

#### `EmptyTrack`

**Location**: tracks.py:560

**Methods**:

- `_plot(self, ax)` (tracks.py:561)

#### `OverlaidTracks`

**Location**: tracks.py:565

**Methods**:

- `__init__(self, tracks_to_overlay, legend)` (tracks.py:566)
- `_plot(self, ax)` (tracks.py:580)

#### `VectorTrack`

**Location**: tracks.py:633

**Methods**:

- `_plot(self, ax)` (tracks.py:646)

#### `VectorTracks`

**Location**: tracks.py:674

**Methods**:

- `_plot(self, ax)` (tracks.py:689)

#### `CutsiteTrack`

**Location**: tracks.py:744

**Methods**:

- `_materialize_data(self)` (tracks.py:752)
- `_plot(self, ax)` (tracks.py:755)

#### `ChipTrack`

**Location**: tracks.py:773

**Description**: This is different from other tracks in that it draws lines which can be overlaid
WiggleTrack fills on the bottom. Also, can normalize this and smooth.

Input is anything that VplotTrack.materialize_numpy_array can take. It will sum along the FL axis
smooth_by: uses smooth1d to smooth with this window size
normalize: normalizes by total depth in region
scale: for scaling al values by a constant. Useful for comparing tracks of different sequencing depths

**Methods**:

- `_materialize_data(self)` (tracks.py:794)
- `_plot(self, ax)` (tracks.py:817)

#### `IntervalTrack`

**Location**: tracks.py:828

**Description**: Plots intervals in a given color. Can add more tracks using +
Input is a 1d numpy array with values indicating positions marked
A peak from position 105 to 109 should be [105, 106, 107, 108]

**Methods**:

- `_materialize_data(self)` (tracks.py:840)
- `_plot(self, ax)` (tracks.py:844)

#### `CpGTrack`

**Location**: tracks.py:851

**Description**: Plots the positions of GpGs in a region. Input is a fasta filepath

**Methods**:

- `_materialize_data(self)` (tracks.py:862)
- `_plot(self, ax)` (tracks.py:870)
- `title(self)` (tracks.py:859)

#### `VplotDiffTrack`

**Location**: tracks.py:875

**Description**: Plots a difference of two Vplots.

**Methods**:

- `_plot(self, ax)` (tracks.py:885)
- `_plot_left(self, ax_main, ax_right)` (tracks.py:908)
- `_plot_right(self, ax_main, ax_right)` (tracks.py:903)

#### `VplotDensityDiffTrack`

**Location**: tracks.py:915

**Methods**:

- `_plot(self, ax)` (tracks.py:923)
- `_plot_left(self, ax_main, ax_right)` (tracks.py:945)
- `_plot_right(self, ax_main, ax_right)` (tracks.py:940)

#### `VplotTrack`

**Location**: tracks.py:952

**Description**: :param input: a str or a list of strs of fnames to stack, or a 2d numpy array, or a FragmentArray

**Methods**:

- `_materialize_data(self)` (tracks.py:978)
- `_plot(self, ax)` (tracks.py:990)
- `_plot_left(self, ax_main, ax_right)` (tracks.py:985)
- `_plot_right(self, ax_main, ax_right)` (tracks.py:981)
- `materialize_numpy_array(input, region)` (tracks.py:1013)
  - Returns a numpy array of a FM from a variety input types (strings, fragment matrix instances, arrays, ...)
- `title(self)` (tracks.py:967)

#### `MotifTrack`

**Location**: tracks.py:1043

**Description**: input is a pwm numpy array

**Methods**:

- `_plot(self, ax)` (tracks.py:1061)
- `from_seqs(seqs)` (tracks.py:1051)

#### `CoverageTrack`

**Location**: tracks.py:1115

**Methods**:

- `_build_coverage(self, strand, fraction, coverage_type)` (tracks.py:1146)
- `_materialize_data(self)` (tracks.py:1137)
- `_plot(self, ax)` (tracks.py:1183)
- `_smooth(self, cov)` (tracks.py:1140)

#### `TfRegionTrack`

**Location**: tracks.py:1247

**Methods**:

- `_plot(self, ax)` (tracks.py:1269)
- `validate_binding_df_scores(self)` (tracks.py:1250)

#### `MultiRegonTfRegionTrack`

**Location**: tracks.py:1303

**Methods**:

- `_plot(self, ax)` (tracks.py:1358)
- `get_tf_cov_arrays(self)` (tracks.py:1308)

#### `MidpointCoverageTrack`

**Location**: tracks.py:1384

#### `EndpointCoverageTrack`

**Location**: tracks.py:1389

#### `SummedEndpointsCoverageTrack`

**Location**: tracks.py:1395

#### `ReadCoverageTrack`

**Location**: tracks.py:1400

**Description**: Input is a BAM file

**Methods**:

- `_materialize_data(self)` (tracks.py:1407)
- `_plot(self, ax)` (tracks.py:1413)

#### `WiggleTrack`

**Location**: tracks.py:1418

**Description**: Input is a wiggle file

**Methods**:

- `_plot(self, ax)` (tracks.py:1445)
- `materialize_numpy_track(input_file, chrom, start, stop, smooth, num_bins)` (tracks.py:1427)

#### `SmallFraction`

**Location**: tracks.py:1467

**Methods**:

- `_materialize_data(self)` (tracks.py:1470)
- `_plot(self, ax)` (tracks.py:1473)

#### `NarrowPeakTrack`

**Location**: tracks.py:1485

**Description**: Input is a narrowpeak bed file

**Methods**:

- `_plot(self, ax)` (tracks.py:1490)

#### `PeakTrack`

**Location**: tracks.py:1509

**Description**: Input is a bed file or list of regions. This is different from BedTrack in that it plots all intervals on the same line

**Methods**:

- `_materialize_data(self)` (tracks.py:1517)
- `_plot(self, ax)` (tracks.py:1530)

#### `BedTrack`

**Location**: tracks.py:1539

**Description**: :param input: a path to a bed file or a list of Regions

**Methods**:

- `_materialize_data(self)` (tracks.py:1548)
- `_plot(self, ax)` (tracks.py:1564)

#### `EpilogosTrack`

**Location**: tracks.py:1602

**Methods**:

- `_materialize_data(self)` (tracks.py:1603)
- `_plot(self, ax)` (tracks.py:1614)
  - Main plotting function. Plots chromosome region to matplotlib axis.

#### `ChromHmmTrack`

**Location**: tracks.py:1700

**Description**: Plots a ChromHMM annotation track with colors
Input is a (num_states, num_positions) matrix, where the values are either one-hot encoded or proportions
If one-hot, each position is plotted as a single ChromHMM state. If proportions, states are stacked.
Resolution can be either the same as the region or coarser, in which case the matrix is aligned
'left', 'right', or 'center'

**Methods**:

- `_materialize_data(self)` (tracks.py:1717)
- `_plot(self, ax)` (tracks.py:1756)

#### `GeneTrack`

**Location**: tracks.py:1832

**Description**: Input is a bed or bed.gz file

**Methods**:

- `_plot(self, ax: axis)` (tracks.py:1839)
  - Main plotting function. Plots chromosome region to matplotlib axis.
- `get_free_row(rows, new_gene)` (tracks.py:1991)
  - This function returns the first row that can be used to plot the gene.

#### `CoverageDifferenceTrack`

**Location**: tracks.py:2011

**Methods**:

- `_build_coverage(self)` (tracks.py:2012)
- `_plot(self, ax)` (tracks.py:2019)

