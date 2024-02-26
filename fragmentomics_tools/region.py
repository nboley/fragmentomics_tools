import os
import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy

import pysam

from fragmentomics_tools.util.dataclass import DataClassMixin
from fragmentomics_tools.constants import DEFAULT_REFERENCE
from fragmentomics_tools.util.liftover import RegionLiftOver
from fragmentomics_tools.contig import (
    CONTIG_LENGTHS,
    CONTIGS,
    STANDARD_CHROMS,
    get_reference_path,
)

from sequence import one_hot_encode_sequences

"""
from fbio.util.dataclass_utils import DataClassMixin
from fbio.util.misc_utils import cmp
from pyDNAbinding.sequence import one_hot_encode_sequences
"""


class OutOfBoundsError(ValueError):
    pass


class ChromOrdering(Enum):
    natural = "natural"
    lexicographical = "lexicographical"


CACHED_READERS = {}
GLOBAL_FASTA_FILE_CACHE = {}


def flip_strand(strand: Optional[str]):
    if strand is None or strand == ".":
        return strand
    else:
        assert strand in {"+", "-"}
        if strand == "+":
            return "-"
        return "+"


@dataclass(repr=False, eq=False)
class Region(DataClassMixin):
    chrom: str
    start: int = None
    stop: int = None
    strand: Optional[str] = None
    ref: str = DEFAULT_REFERENCE
    data: dict = None

    @property
    def name(self):
        return self.data["name"] if self.data else None

    def flip_strand(self):
        return self.replace(strand=flip_strand(self.strand))

    def shift(self, offset: int = 0):
        self.replace(start=self.start + offset, stop=self.stop + offset)

    def intersect_annotation(self, annotation_name: str) -> List:
        """
        Intersects this Region with a annotation_name

        :param annotation_name: ex: 'repeat_masker' or 'anshul_blacklist'.  See fbio.annotation.ANNOTATIONS.keys()
          for options.
        :return: A List of Records

        >>> list(Region('chr1',10000,10010).intersect_annotation('repeat_masker'))
        [Bed6Record(chrom='chr1', start=10000, stop=10468, name='(TAACCC)n', score=463.0, strand='+')]
        >>> list(Region('chr1',10000,10010).intersect_annotation('anshul_blacklist'))
        [Bed3Record(chrom='chr1', start=0, stop=792500)]
        >>> list(Region('chr1',10000,10010).intersect_annotation('not-a-track'))
        Traceback (most recent call last):
        ....
        KeyError: 'Could not find annotation for name `not-a-track` and reference `hg38`'
        """
        from fbio.annotations import ANNOTATIONS

        if (annotation_name, self.ref) not in CACHED_READERS:
            CACHED_READERS[(annotation_name, self.ref)] = ANNOTATIONS.get(
                annotation_name, self.ref
            ).get_reader()
        for record in CACHED_READERS[(annotation_name, self.ref)].fetch(
            self.chrom, self.start, self.stop
        ):
            yield record

    def get_coverage_array(self, regions) -> numpy.ndarray:
        """
        Gets a coverage array of the number of pileup intersections with the regions in regions.
         Similar to the bedtools coverage coverage command.

        :param regions: a List of objects which have a chrom, start, and stop attribute

        >>> regions = [Region('chr1', 100, 110), Region('chr1', 105, 115)]
        >>> Region('chr1', 105, 107).get_coverage_array(regions)
        array([2, 2])
        >>> Region('chr1', 98, 100).get_coverage_array(regions)
        array([0, 0])
        >>> Region('chr1', 99, 106).get_coverage_array(regions)
        array([0, 1, 1, 1, 1, 1, 2])
        >>> Region('chr1', 99, 106).get_coverage_array(regions + [Region('chr2', 0, 1)])
        Traceback (most recent call last):
        ...
        ValueError: chr2 != chr1
        """
        if isinstance(regions, Region):
            regions = [regions]

        arr = numpy.zeros(self.length, dtype=int)
        for region in regions:
            if region.chrom != self.chrom:
                raise ValueError(f"{region.chrom} != {self.chrom}")

            arr[
                max(region.start - self.start, 0) : min(
                    region.stop - self.start, self.stop
                )
            ] += 1
        return arr

    def get_annotation_coverage_array(self, annotation_names) -> numpy.ndarray:
        """
        :param annotation_names: ex: ['repeat_masker'] one of fbio.annotation.ANNOTATION.keys()
        :returns:
        """
        if isinstance(annotation_names, str):
            annotation_names = [annotation_names]

        return sum(
            self.get_coverage_array(self.intersect_annotation(annotation_name))
            for annotation_name in annotation_names
        )

    def get_overlaps_repeat_or_blacklist_mask_array(self):
        """
        :return: A boolean array which is True everywhere the region intersects a repeat element or an element
          in Anshul's blacklist.

        >>> is_repeat = Region('chr1',794555-5,794555+5).get_overlaps_repeat_or_blacklist_mask_array()
        >>> is_repeat
        array([False, False, False, False, False,  True,  True,  True,  True,  True])
        >>> is_repeat.mean()
        0.5
        """
        return self.get_annotation_coverage_array(
            ["repeat_masker", "anshul_blacklist"]
        ).astype(bool)

    def intersect_with_bed(self, bed_path):
        # Prevents circular import
        from fbio.formats import TabixBedReader

        if bed_path not in CACHED_READERS:
            # Create a new reader
            CACHED_READERS[(bed_path, self.ref)] = TabixBedReader(bed_path)
        for record in CACHED_READERS[(bed_path, self.ref)].fetch(
            self.chrom, self.start, self.stop
        ):
            yield record

    def get_bed_coverage_array(self, bed_path) -> numpy.ndarray:
        """
        :param bed_path:
        :returns:
        """
        return self.get_coverage_array(self.intersect_with_bed(bed_path))

    def plot_annotations(self, annotation_names: List[str] = None):
        """
        Plots annotation tracks over this region

        :param annotation_names: a list of annotation names from fbio.annotations.ANNOTATIONS

        >>> _ = Region('chr1', 10000, 10100).plot_annotations()
        """
        from fbio.plot.tracks import BedTrack, Tracks
        from fbio.annotations import ANNOTATIONS

        if annotation_names is None:
            annotation_names = ["repeat_masker", "anshul_blacklist"]

        tracks = []
        for annotation_name in annotation_names:
            annotation = ANNOTATIONS.get(annotation_name, self.ref)
            tracks.append(
                BedTrack(annotation.local_path, region=self, name=annotation.name)
            )

        return Tracks(tracks).plot()

    def convert_region(self, unique_liftoverer: RegionLiftOver):
        """Returns a copy of converted into the new reference."""
        if not isinstance(unique_liftoverer, RegionLiftOver):
            raise TypeError(
                "Expecting an instance of RegionLiftOver for unique_liftoverer "
                "(received {})".format(type(unique_liftoverer))
            )
        if unique_liftoverer.source != self.ref:
            raise ValueError(
                "Attempting to convert region with ref {} from {} to {}".format(
                    self.ref, unique_liftoverer.source, unique_liftoverer.dest
                )
            )
        res = unique_liftoverer.uniquely_convert_region(
            self.chrom, self.start, self.stop
        )
        # check that this region can be converted
        if res is None:
            return None
        else:
            chrom, start, stop, strand = res

        return type(self)(
            chrom,
            start,
            stop,
            strand=strand,
            ref=unique_liftoverer.dest,
            data=self.data,
        )

    def liftover(self, *args, **kwargs):
        """Alias for convert_region."""
        return self.convert_region(*args, **kwargs)

    @property
    def midpoint(self):
        return self.start + self.length // 2

    @staticmethod
    def get_resize_start(
        start: int, current_size: int, new_size: int, strand: Optional[str] = None
    ):
        midpoint = start + current_size // 2
        if strand == "-" and current_size % 2 == 0 and new_size % 2 == 1:
            # Resizing from even to odd. Extending 1 space further to the left (5' for negative strand)
            new_start = midpoint - (new_size // 2) - 1
        elif strand == "-" and current_size % 2 == 1 and new_size % 2 == 0:
            # Resizing from odd to even.Extending 1 space further to the RIGHT (3' for negative strand)
            new_start = midpoint - (new_size // 2) + 1
        else:
            new_start = midpoint - new_size // 2

        return new_start

    @staticmethod
    def get_resize_starts(
        start: numpy.ndarray,
        current_size: numpy.ndarray,
        new_size: numpy.ndarray,
        strand: Optional[numpy.ndarray] = None,
    ):
        # This should work on vectors as well
        midpoint = start + current_size // 2
        new_start = midpoint - new_size // 2

        if strand is not None:
            # Resizing from even to odd. Extending 1 space further to the left (5' for negative strand)
            new_start[
                (strand == "-")
                & (numpy.mod(new_size, 2) == 1)
                & (numpy.mod(current_size, 2) == 0)
            ] -= 1
            # Resizing from odd to even.Extending 1 space further to the RIGHT (3' for negative strand)
            new_start[
                (strand == "-")
                & (numpy.mod(new_size, 2) == 0)
                & (numpy.mod(current_size, 2) == 1)
            ] += 1

        return new_start

    def resize(self, new_size):
        """
        You can think about resizing as an iterative process
        If increasing a region, you keep adding one base pair at a time, alternating right and left
        Whether you add to the right or left *first* depends on the parity of the region length
        If the region starts off even length, you add to the right first, then left, then right...
        This is the opposite with an odd length. Add on the left, right, left

        With decreasing region size, it should be the *inverse* of the above
        That means that if you have an even length region, you remove from the left, then right, left
        For odd length, remove right, then left, right

        However, because of motifs, this should all be reversed if you start on the negative strand
        So if a motif of one parity is resized to a different parity, positive and negative strand are not
        consistent with the above schema. Therefore, if you have a region on the negative strand:
        even -> odd region, increasing: left, right, left
        odd -> even region, increasing: : right, left, right
        even -> odd region, decreasing: right, left, right
        odd -> even region, decreasing: : left, right, left

        Thankfully, this only matters if the region changes parity, so the math is much simpler in most cases
        """
        if new_size is None or new_size == self.length:
            # The user asked for the same size, or is just using this as a copy
            return self.replace()

        new_start = self.get_resize_start(
            start=self.start,
            current_size=self.length,
            new_size=new_size,
            strand=self.strand,
        )

        if new_start < 0:
            raise OutOfBoundsError(
                f"There is not enough flanking sequence to center this region "
                f"(would result in a start coordinate of '{new_start}')"
            )

        new_stop = new_start + new_size
        if self.chrom != "NA" and new_stop > CONTIG_LENGTHS[self.ref][self.chrom]:
            raise OutOfBoundsError(
                f"There is not enough flanking sequence to center this region "
                f"(would result in a stop coordinate of '{new_stop}' but the "
                f"chrom length is '{CONTIG_LENGTHS[self.chrom]}')"
            )

        rv = type(self)(
            self.chrom, new_start, new_stop, self.strand, self.ref, self.data
        )
        # assert rv.midpoint == self.midpoint, f"New mp: {self.midpoint} VS Old mp: {rv.midpoint}"

        return rv

    @staticmethod
    def parse_region_str(s):
        """
        Parses a string into chrom, start, stop, strand

        >>> Region.parse_region_str('chr1:+:100-200')
        ('chr1', '+', 100, 200)
        >>> Region.parse_region_str('chr1:100-200')
        ('chr1', None, 100, 200)
        >>> Region.parse_region_str('chr1	100	200')
        ('chr1', None, 100, 200)
        >>> Region.parse_region_str('chr1 100 200')
        ('chr1', None, 100, 200)
        """
        # split spaces/tabs
        parts = re.split(r"[\s\t]+", s.strip())
        if len(parts) > 1:
            assert (
                len(parts) == 3
            ), "if there is a tab is the region string, it must contain three parts"
            return parts[0], None, int(parts[1]), int(parts[2])
        else:
            data = s.strip().split(":")
            if len(data) == 3:
                chrom, strand, start_and_stop = data
                if strand not in "+-.":
                    raise ValueError(f"Unrecognized strand '{strand}'")
            elif len(data) == 2:
                chrom, start_and_stop = data
                strand = None
            else:
                raise ValueError(f"Could not parse region str '{s}'")

            start, stop = start_and_stop.split("-")

            return chrom, strand, int(start), int(stop)

    @classmethod
    def from_region_str(cls, region_str, ref: str = DEFAULT_REFERENCE, data=None):
        chrom, strand, start, stop = Region.parse_region_str(region_str)
        return cls(
            chrom=chrom, start=start, stop=stop, strand=strand, ref=ref, data=data
        )

    def __post_init__(self):
        """
        >>> Region('chr2', 1, 2)
        Region(chr2:1-2)
        >>> Region('chr2', 1, None)
        Region(chr2:1-242193529)
        >>> Region('chr2:10000000-11000000')
        Region(chr2:10000000-11000000)
        """
        assert isinstance(self.chrom, str)

        if self.ref not in CONTIG_LENGTHS:
            raise ValueError(f"{self.ref} is not in CONTIG_LENGTHS")

        if ":" in self.chrom or "\t" in self.chrom:
            self.chrom, self.strand, self.start, self.stop = Region.parse_region_str(
                self.chrom
            )

        if self.chrom != "NA" and self.chrom not in CONTIG_LENGTHS[self.ref]:
            raise ValueError(f"{self.chrom} not found in CONTIG_LENGTHS for {self.ref}")

        assert (
            self.data is None or "strand" not in self.data
        ), "'strand' should now be passed into init"

        if self.start is None:
            self.start = 0
        if self.stop is None:
            if self.chrom == "NA":
                raise ValueError("cannot infer stop when chrom is NA")
            self.stop = CONTIG_LENGTHS[self.ref][self.chrom]
        if self.strand == ".":
            self.strand = None

        # Sometimes, we want to plot a region with negative coordinates, like relative to some landmark
        # In these situations, we have a "NA" chromosome and coordinates that go from -x to x
        assert self.chrom == "NA" or self.start >= 0

        if self.chrom != "NA" and self.stop > CONTIG_LENGTHS[self.ref][self.chrom]:
            raise ValueError(
                f"stop ({self.stop}) is greater than {self.chrom} length ({CONTIG_LENGTHS[self.ref][self.chrom]})"
            )
        if self.start > self.stop:
            raise ValueError(
                f"stop ({self.stop}) can not be less than start ({self.start})"
            )
        if self.strand not in [None, "+", "-"]:
            raise ValueError(
                f"strand='{self.strand}' expected to be one of: [None, '+', '-']"
            )

    @property
    def length(self):
        return self.stop - self.start

    def intersects(self, other: "Region") -> bool:
        """
        >>> x = Region('chr1', 10, 20)
        >>> y = Region('chr1', 5, 15)
        >>> z = Region('chr1', 20, 40)
        >>> x.intersects(y)
        True
        >>> y.intersects(x)
        True
        >>> x.intersects(z)
        False
        """
        return regions_intersect(
            self.chrom, self.start, self.stop, other.chrom, other.start, other.stop
        )

    def cmp(self, region, chrom_ordering=ChromOrdering.natural):
        """
        If regions intersect, return 0
        if self < region, return -1
        if self > region, return 1

        >>> Region('chr1', 40, 50).cmp(Region('chr1', 50, 60))
        -1
        >>> Region('chr1', 40, 51).cmp(Region('chr1', 50, 60))
        0
        >>> Region('chr1', 60, 61).cmp(Region('chr1', 50, 60))
        1
        >>> Region('chr2', 40, 51).cmp(Region('chr1', 50, 60))
        1
        """
        if self.chrom == region.chrom:
            c = 0
        else:
            if chrom_ordering == ChromOrdering.lexicographical:
                c = cmp(self.chrom, region.chrom)
            elif chrom_ordering == ChromOrdering.natural:
                c = cmp(
                    CONTIGS[self.ref].index(self.chrom),
                    CONTIGS[self.ref].index(region.chrom),
                )
            else:
                raise AssertionError("impossible")

        if c != 0:
            return c
        else:
            if self.intersects(region):
                return 0
            elif self.stop <= region.start:
                return -1
            elif self.start >= region.stop:
                return 1
            else:
                raise AssertionError("Impossible")

    def __lt__(self, other):
        """
        >>> Region('chr1', 100, 200) < Region('chr1', 200, 300)
        True
        >>> Region('chr1', 100, 200) < Region('chr1', 199, 300)
        False
        """
        assert isinstance(other, Region), "invalid type comparison"
        return self.cmp(other) == -1

    def __gt__(self, other):
        """
        >>> Region('chr1', 200, 300) > Region('chr1', 100, 200)
        True
        """
        assert isinstance(other, Region), "invalid type comparison"
        return self.cmp(other) == 1

    def __ne__(self, other):
        """
        >>> Region('chr1', 100, 200) != Region('chr1', 100, 201)
        True
        """
        return not self == other

    def __repr__(self):
        return f"Region({self.__str__()})"

    def __str__(self):
        if self.strand_is_set():
            return f"{self.chrom}:{self.strand}:{self.start}-{self.stop}"
        else:
            return f"{self.chrom}:{self.start}-{self.stop}"

    def __eq__(self, other):
        """
        >>> Region('chr1', 1, 2) == Region('chr1', 1, 2)
        True
        >>> Region('chr1', 1, 2) == Region('chr1', 2, 3)
        False
        >>> Region('chr1', 1, 2, '+') == Region('chr1', 1, 2, '+')
        True
        >>> Region('chr1', 1, 2, '+') == Region('chr1', 1, 2, '-')
        False
        >>> Region('chr1', 100, 200, data={'name': 1}) == Region('chr1', 100, 200, data={'name': 2})
        False
        >>> Region('chr1', 100, 200, data={'namea': 1}) == Region('chr1', 100, 200, data={'name': 1})
        False
        """
        assert isinstance(other, Region) or other.__class__.__name__ == "Region"
        return all(
            getattr(self, f.name) == getattr(other, f.name) for f in fields(self)
        )

    def __contains__(self, other):
        """
        Tests whether one region is a subregion of another
        >>> Region("chr1", 120, 180) in Region("chr1", 100, 200)
        True
        >>> Region("chr1", 100, 200) in Region("chr1", 120, 180)
        False
        >>> Region("chr2", 120, 180) in Region("chr1", 100, 200)
        False
        >>> Region("chr1", 120, 200) in Region("chr1", 100, 200)
        True
        >>> Region("chr1", 120, 201) in Region("chr1", 100, 200)
        False
        """
        assert isinstance(other, Region), "invalid type comparison"
        return (
            (other.chrom == self.chrom)
            and (self.start <= other.start)
            and (self.stop >= other.stop)
        )

    def __hash__(self):
        return hash(
            (
                self.chrom,
                self.start,
                self.stop,
                self.strand,
                self.ref,
                (None if self.data is None else tuple(self.data.items())),
            )
        )

    def to_bed3_line(self):
        """contig start stop"""
        return f"{self.chrom}\t{self.start}\t{self.stop}"

    @property
    def bed_strand(self):
        """Return strand if strand in '+-.' and '.' if None"""
        if self.strand is None:
            return "."
        assert self.strand in ".+-"
        return self.strand

    def to_bed6_line(self, name=".", score=1000):
        """contig start stop"""
        return f"{self.chrom}\t{self.start}\t{self.stop}\t{name}\t{score}\t{self.bed_strand}"

    @classmethod
    def random(cls, length, assembly, chroms=None, strand=None) -> "Region":
        """
        :param length: length of the region
        :param assembly: 'hg38' or 'hg19'
        :param chroms: chromosomes to choose from, defaults to STANDARD_CHROMS
        :param strand: strand
        :return: a random region

        >>> numpy.random.seed(1)
        >>> Region.random(100, 'hg38')
        Region(chr6:140980108-140980208)
        >>> Region.random(10, 'hg38')
        Region(chr9:491263-491273)
        """
        if chroms is None:
            chroms = STANDARD_CHROMS

        chrom = numpy.random.choice(chroms, 1)[0]
        start = numpy.random.randint(0, CONTIG_LENGTHS[assembly][chrom] - length)
        stop = start + length

        return cls(chrom, start, stop, strand=strand)

    def shift(self, n) -> "Region":
        """
        Shifts a region by n bases.  Use a negative to shift upstream, positive to shift it downstream
        :param n: number of bases to shift by
        :return: A shifted Region

        >>> Region('chr1', 100, 200).shift(-10)
        Region(chr1:90-190)
        >>> Region('chr1', 100, 200).shift(10)
        Region(chr1:110-210)
        """
        return self.replace(start=self.start + n, stop=self.stop + n)

    def strand_is_set(self) -> bool:
        """Return true if the strand is set to something, also verify it is valid."""
        strand_set = self.strand is not None and self.strand != "."
        if strand_set:
            assert self.strand in {"+", "-"}, f"Invalid strand {self.strand}"
        return strand_set

    def is_minus_strand(self) -> bool:
        """Return True if the strand is set and it is -"""
        return self.strand_is_set() and self.strand == "-"

    def get_sequence(
        self,
        reference_fasta: pysam.Fastafile,
        reverse_complement_sequence_if_minus_strand: bool = False,
    ) -> numpy.ndarray:
        """
        get the sequence of a region from a fasta file
        :param reverse_complement_sequence_if_minus_strand: If set to True, then flip/RC the sequence if it is on the minus strand.
        :return: a 4xlength one-hot encoded numpy array
        """
        seq = one_hot_encode_sequences(
            [reference_fasta.fetch(self.chrom, self.start, self.stop).encode()]
        ).astype("int8")
        one_hot = seq[0]
        if reverse_complement_sequence_if_minus_strand and self.is_minus_strand():
            return one_hot[::-1, ::-1]
        else:
            return one_hot

    def get_subregion_from_jitter_and_strand(
        self, output_size: int, jitter: int, strand: Optional[str] = None
    ):
        new_resize_start = self.get_resize_start(
            start=self.start,
            current_size=self.length,
            new_size=output_size,
            strand=strand,
        )
        new_jittered_start = new_resize_start + jitter
        new_stop = new_jittered_start + output_size
        return type(self)(
            self.chrom, new_jittered_start, new_stop, strand, self.ref, self.data
        )


def intervals_intersect(x_start, x_stop, y_start, y_stop):
    """Test whether two closed-open intervals intersect.

    >>> intervals_intersect(0,1, 0,1)
    True
    >>> intervals_intersect(0,1, 1,2)
    False
    >>> intervals_intersect(0,1, 0,0)
    False
    >>> intervals_intersect(1,2, 0,1)
    False
    >>> intervals_intersect(1,2, 0,2)
    True
    """
    return x_start < y_stop and y_start < x_stop


def intervals_intersect_with_none(x_start, x_stop, y_start, y_stop):
    """
    >>> intervals_intersect_with_none(0, 1, None, 2)
    True
    >>> intervals_intersect_with_none(0, 1, None, None)
    True
    >>> intervals_intersect_with_none(None, 2, 2, 4)
    False
    >>> intervals_intersect_with_none(None, None, 2, 4)
    True
    >>> intervals_intersect_with_none(2, 4, 4, None)
    False
    """
    if x_start is None:
        x_start = 0
    if y_start is None:
        y_start = 0
    if x_stop is None:
        x_stop = float("inf")
    if y_stop is None:
        y_stop = float("inf")

    return intervals_intersect(x_start, x_stop, y_start, y_stop)


def regions_intersect(x_chrom, x_start, x_stop, y_chrom, y_start, y_stop):
    assert x_chrom is not None
    assert y_chrom is not None
    return x_chrom == y_chrom and intervals_intersect(x_start, x_stop, y_start, y_stop)


def region_to_region_str(chrom=None, start=None, stop=None):
    """
    >>> region_to_region_str('chr1')
    'chr1'
    >>> region_to_region_str('chr1', 100)
    'chr1:100'
    >>> region_to_region_str('chr1', 100, 200)
    'chr1:100-200'
    """
    s = ""
    if chrom is not None:
        s += f"{chrom}"
    if start is not None:
        s += f":{start}"
    if stop is not None:
        s += f"-{stop}"
    return s
