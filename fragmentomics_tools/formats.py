import abc
import bisect
import dataclasses
import importlib
import inspect
import io
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from collections import defaultdict
from dataclasses import fields, dataclass, InitVar
from itertools import islice, groupby
from tempfile import TemporaryDirectory
from typing import Iterable, List, Union, Optional
import string
import random

import numpy
import pandas
import pyBigWig
import pysam
import smart_open
from fragmentomics_tools.region import region_to_region_str, intervals_intersect_with_none, Region

from smart_open import open

# TODO -- move this into config
DEFAULT_FBIO_CACHE_DIR = os.environ.get("DEFAULT_FBIO_CACHE_DIR", "/ssd/fbio_cache")

log = logging.getLogger(__name__)

def _bash_iter(
    cmd, print_lines: Union[bool, TextIO] = False, print_cmd: Union[bool, TextIO] = False,
):
    assert not cmd.startswith("rm -rf /")
    cmd = f"set -eo pipefail && {cmd}"

    def print_if_flag(s, flag):
        if flag is False:
            pass
        elif flag is True:
            print(s)
        elif hasattr(flag, "write"):
            print(s, file=flag)
        else:
            raise ValueError(f"{flag} must be a bool or a file")

    if print_cmd:
        print_if_flag("+ " + cmd, print_cmd)
    p = sp.Popen(cmd, shell=True, executable="/bin/bash", stdout=sp.PIPE, stderr=sp.PIPE)

    output = []
    for line in p.stdout:
        line = line.decode().rstrip("\n")
        print_if_flag(line, print_lines)
        output.append(line)
        yield line
    p.wait()
    if p.returncode != 0:
        raise VerboseCalledProcessError(p.returncode, cmd, "\n".join(output[:100]), p.stderr.read().decode())


def _bash(
    cmd, print_lines: Union[bool, TextIO] = False, print_cmd: Union[bool, TextIO] = False,
):
    return list(_bash_iter(cmd, print_lines=print_lines, print_cmd=print_cmd))


def _get_subclasses_of(cls, include_self=False):
    subclasses = set(cls.__subclasses__()).union(
        s for c in cls.__subclasses__() for s in get_subclasses_of(c)
    )
    return subclasses.union({cls}) if include_self else subclasses

def _windowed_range(start, stop, window_size):
    """
    >>> list(windowed_range(0, 5, 2))
    [(0, 2), (2, 4), (4, 5)]
    >>> list(windowed_range(0, 1, 2))
    [(0, 1)]
    >>> list(windowed_range(0, 11, 3))
    [(0, 3), (3, 6), (6, 9), (9, 11)]
    >>> list(windowed_range(-3, 3, 3))
    [(-3, 0), (0, 3)]
    """
    if window_size <= 0:
        raise ValueError("invalid window size")
    if stop <= start:
        raise ValueError("invalid start/stop")

    for start in range(start, stop, window_size):
        yield start, min(stop, start + window_size)

def _is_s3_uri(s):
    return isinstance(s, str) and s.startswith("s3://")


def _get_file_path_and_file_hash_fast(path, n_bytes=1024, url_ok=False):
    """
    get a file hash based on the path, the file size, and the first n_bytes

    :param bool url_ok: if path is a url, skip using the first n_bytes and file size for hashing
    """
    m = hashlib.md5()
    m.update(path.encode())
    if (path.startswith("http") or path.startswith("s3")) and url_ok:
        # if path is a url, don't use size or content
        # TODO get info from http/s3 headers?
        return m.hexdigest()
    else:
        # use file size
        m.update(str(os.stat(path).st_size).encode())
        with open(path, "rb") as fp:
            m.update(fp.read(n_bytes))

        return m.hexdigest()


@dataclass
class Fragment:
    __slots__ = [
        "chrom",
        "start",
        "stop",
        "mapq1",
        "mapq2",
        "gc",
        "strand",
        "cell_barcode",
        "num_cpgs",
        "num_meth_cpgs",
    ]

    def __init__(
        self,
        chrom: str,
        start: int,
        stop: int,
        mapq1: Optional[int] = None,
        mapq2: Optional[int] = None,
        gc: Optional[float] = None,
        strand: Optional[str] = None,
        cell_barcode: Optional[str] = None,
        num_cpgs: Optional[int] = None,
        num_meth_cpgs: Optional[int] = None,
    ):
        assert isinstance(start, (int, numpy.int64, numpy.int32))
        if strand in [b"+", b"-"]:
            strand = strand.decode()
        assert strand in [None, "+", "-"]
        for k, v in locals().items():
            # @py_assert2 check is to avoid a pytest error introduced by it changing state
            if k != "self" and not k.startswith("@py_assert"):
                setattr(self, k, v)

    def mapq_gte(self, threshold):
        """
        >>> Fragment('chr1', 0, 1, mapq1=9).mapq_gte(10)
        False
        >>> Fragment('chr1', 0, 1, mapq2=10).mapq_gte(10)
        True

        None mapqs always pass
        >>> Fragment('chr1', 0, 1).mapq_gte(90)
        True
        """
        return self.mapq12_min is None or self.mapq12_min >= threshold

    @property
    def mapq12_min(self):
        """Returns the minimum of self.mapq1 and self.mapq2 ignoring None values
        >>> Fragment('chr1', 0, 1, mapq1=1, mapq2=0).mapq12_min
        0
        >>> Fragment('chr1', 0, 1, mapq1=0, mapq2=0).mapq12_min
        0
        >>> Fragment('chr1', 0, 1, mapq1=None, mapq2=2).mapq12_min
        2
        >>> Fragment('chr1', 0, 1, mapq1=1, mapq2=None).mapq12_min
        1
        >>> Fragment('chr1', 0, 1, mapq1=None, mapq2=None).mapq12_min is None
        True
        """

        def none_to_inf(x):
            return float("inf") if x is None else x

        if self.mapq1 is None and self.mapq2 is None:
            return None
        elif self.mapq1 is None and self.mapq2 is not None:
            return self.mapq2
        elif self.mapq1 is not None and self.mapq2 is None:
            return self.mapq1
        else:
            return min(self.mapq1, self.mapq2)

    def replace(self, **kwargs):
        d = {field_name: getattr(self, field_name) for field_name in self.field_names}
        d.update(**kwargs)
        return self.__class__(**d)

    @classproperty
    def field_names(cls):
        return cls.__slots__

    @classproperty
    def field_types(cls):
        sig = funcsigs.signature(cls.__init__)
        return [sig.parameters[field].annotation for field in cls.field_names]

    def to_line(self):
        def none_to_empty_string(val):
            if val is None:
                return ""
            else:
                return val

        mapq1 = none_to_empty_string(self.mapq1)
        mapq2 = none_to_empty_string(self.mapq2)
        gc = none_to_empty_string(self.gc)
        strand = none_to_empty_string(self.strand)

        # extra fields must be comma separated in the last column, otherwise bedToBigBed fails
        return f"{self.chrom}\t{self.start}\t{self.stop}\t{mapq1},{mapq2},{gc},{strand}"

    @classmethod
    def from_line_parts(cls, parts):
        def cast(s, type_):
            if s in ("", None):
                return None
            else:
                return type_(s)

        assert len(parts) == 4
        contig, start, stop, attrs = parts
        attr_parts = attrs.split(",")

        mapq1 = cast(attr_parts[0], int)
        strand = None
        # if the length of attr_parts, try to cast the last one
        # to an int. If that works, then assume they are mapqs.
        # Otherwise, assume that it is GC.
        if len(attr_parts) == 1:
            mapq2 = None
            gc = None
        elif len(attr_parts) == 2:
            try:
                cast(attr_parts[1], int)
            except ValueError:
                mapq2 = mapq1
                gc = cast(attr_parts[1], float)
            else:
                mapq2 = cast(attr_parts[1], int)
                gc = None
        elif len(attr_parts) >= 3:
            mapq2 = cast(attr_parts[1], int)
            gc = cast(attr_parts[2], float)
            if len(attr_parts) == 4 and attr_parts[3] in ["+", "-"]:
                strand = attr_parts[3]

        return cls(contig, int(start), int(stop), mapq1, mapq2, gc, strand)

    @classmethod
    def from_line(cls, line):
        return cls.from_line_parts(line.rstrip("\n").split("\t"))

    @property
    def tlen(self):
        return self.stop - self.start

    @property
    def length(self):
        return self.tlen

    @property
    def midpoint(self):
        return self.start + (self.stop - self.start) // 2

    def __repr__(self):
        items = ",".join(
            "%s=%r" % (k, getattr(self, k)) for k in self.field_names if getattr(self, k) is not None
        )
        return f"Fragment({items})"

    def __eq__(self, other):
        return [getattr(self, k) for k in self.field_names] == [getattr(other, k) for k in self.field_names]

    @staticmethod
    def length_and_midpoint_to_start_and_stop(length, midpoint):
        """
        Converts length and midpoint representation to start and stop representation.  Note that there is a
        1:1 mapping for any (length, midpoint) to (start, stop).  Midpoint is defined as the floor of the exact middle
        of (start, stop).
        >>> Fragment.length_and_midpoint_to_start_and_stop(10, 5)
        (0, 10)
        >>> Fragment.length_and_midpoint_to_start_and_stop(9, 5)
        (1, 10)

        """
        start = midpoint - length // 2
        # add one to the stop if length is odd, since the actual midpoint was midpoint + .5
        stop = midpoint + length // 2 + (length % 2)

        if isinstance(start, (int, numpy.int64, numpy.int32)):
            # start/stop are scalars
            assert start < stop, f"impossible coordinates: {start}, {stop}"
        elif isinstance(start, numpy.ndarray):
            # start/stop are arrays
            assert (start < stop).all(), f"impossible coordinates: {start}, {stop}"
        else:
            raise TypeError(f"{start} is not a valid dtype")

        return start, stop

    @classmethod
    def from_bed12_parts(cls, parts):
        assert len(parts) == 12
        if parts[10] != ".":
            methyls = parts[10].split(",")
            num_cpgs = len(methyls)
            num_meth_cpgs = int(numpy.array([methyl == "1" for methyl in methyls]).sum())
        else:
            num_cpgs = 0
            num_meth_cpgs = 0

        return cls(
            chrom=parts[0],
            start=int(parts[1]),
            stop=int(parts[2]),
            strand=parts[5],
            num_cpgs=num_cpgs,
            num_meth_cpgs=num_meth_cpgs,
        )


@dataclass
class FastqRecord:
    qname: str
    seq: str
    extra_line: str
    qual: str


class FastqReader:
    def __init__(self, in_fname):
        self.in_fname = in_fname
        self.reader = smart_open.open(self.in_fname)

    def __next__(self) -> FastqRecord:
        data = list(islice(self.reader, 0, 4))
        if len(data) == 0:
            raise StopIteration
        else:
            return FastqRecord(*[line.rstrip("\n") for line in data])

    def __iter__(self) -> Iterable[FastqRecord]:
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()


class RegionRecord:
    chrom: str
    start: int
    stop: int
    strict: InitVar[bool] = True

    @property
    def length(self):
        return self.stop - self.start

    @property
    def region(self) -> Region:
        """A Region object represted by this record"""
        return Region(self.chrom, self.start, self.stop, strand=getattr(self, "strand", None))

    def __post_init__(self, strict):
        """
        Fields can either be:
         * The correct type
         * 'NA' or None
         * a string which can be converted to the correct type, without loss of information
            (ex "2.2" is not a valid int)

        :param strict:
        """
        # FIXME should probably move any casting from lines or strings into from_line()/from_line_parts()?
        # If possible, cast to the annotation type
        for field in fields(self):
            raw_val = getattr(self, field.name)
            if raw_val in [".", None]:
                setattr(self, field.name, None)
            elif isinstance(raw_val, field.type):
                # type is what we want
                setattr(self, field.name, raw_val)
            else:
                if not isinstance(raw_val, str):
                    raise TypeError(f"{field.name}'s value of `{raw_val}` is not type {field.type}")
                try:
                    new_val = field.type(raw_val)
                    setattr(self, field.name, new_val)
                except ValueError:
                    if strict:
                        raise ValueError(
                            f"Could not cast {field.name}'s value of `{raw_val}` to type {field.type}"
                        )

                if strict and raw_val != str(new_val):
                    raise ValueError(
                        f"Cannot safely change {field.name}'s value of `{raw_val}` to type {field.type}, "
                        f"new value would be {new_val}"
                    )

            if field.name == "strand":
                if getattr(self, "strand") not in [None, ".", "", "+", "-"]:
                    raise ValueError(f'invalid strand: "{self.strand}".')

    @classmethod
    def from_line_parts(cls, parts: List[str]):
        return cls(*parts)

    @classmethod
    def from_line(cls, line: str):
        assert not line.endswith("\n"), f"Invalid: line ends with newline: {line}"
        return cls.from_line_parts(line.split("\t"))

    def to_line(self):
        def encode(val):
            if val is None:
                return "."
            else:
                return str(val)

        return "\t".join(encode(getattr(self, field.name)) for field in fields(self))

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


@dataclass
class Bed3Record(RegionRecord):
    chrom: str
    start: int
    stop: int
    strict: InitVar[bool] = False


@dataclass
class Bed4Record(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    strict: InitVar[bool] = False


@dataclass
class Bed5Record(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    score: float
    strict: InitVar[bool] = False


@dataclass
class Bed6Record(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    score: float
    strand: str
    strict: InitVar[bool] = False


@dataclass
class Bed12Record(RegionRecord):
    # Use this for methyl beds
    chrom: str
    start: int
    stop: int
    name: str
    score: int  # Percent methylation (out of 1000)
    strand: str
    thickStart: str
    thickEnd: str
    itemRgb: str  # Red: 100% methylated, blue: unmethylated
    blockCount: int  # number of CpGs
    blockSizes: str  # boolean (0, 1) indicating methylation
    blockStarts: str  # The positions where we have a CpG
    strict: InitVar[bool] = False


@dataclass
class GappedPeakRecord(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    score: int
    strand: str
    thick_start: int
    thick_end: int
    item_rgb: int
    block_count: int
    signal_value: int
    p_value: float
    q_value: float
    peak: int
    strict: InitVar[bool] = False


@dataclass
class WigRecord(RegionRecord):
    chrom: str
    start: int
    stop: int
    score: float
    strict: InitVar[bool] = False

    @classmethod
    def iter_wiggle_records_from_region_scores(
        cls,
        scores: Union[list, numpy.ndarray],
        region: Region,
        min_score: float = None,
        max_score: float = None,
    ):
        """
        Converts a numpy array to wiggle records by grouping identical scores into a single region.
        nans do not produce wiggle records.

        :param scores: numpy array of scores
        :param region: region the array represents
        :param min_score: minimum score to report in the wiggle records
        :param max_score: maximum score to report in the wiggle records

        >>> region = Region('chr1', 500, 506)
        >>> arr = [1.,2.,2.,numpy.nan,6.,6.]
        >>> list(WigRecord.iter_wiggle_records_from_region_scores(arr, region, min_score=2, max_score=5))
        [WigRecord(chrom='chr1', start=501, stop=503, score=2.0)]
        """
        scores = numpy.asarray(scores).copy()

        if region.length != len(scores):
            raise ValueError(f"score length does not equal region length.  {len(scores)} != {region.length}")
        if isinstance(scores, numpy.ndarray) and scores.ndim != 1:
            raise ValueError(f"score must be a vector. score has {scores.ndim} dimensions")

        # errstate is to ignore nan values that are in scores
        with numpy.errstate(invalid="ignore"):
            if min_score is not None:
                scores[scores < min_score] = numpy.nan
            if max_score is not None:
                scores[scores > max_score] = numpy.nan

        # group identical scores together, and then yield a WiggleRecord of that score over the relevant
        # position
        for score, positions in groupby(zip(range(region.start, region.stop), scores), lambda t: t[1]):
            if numpy.isnan(score):
                continue
            positions, score_group = list(zip(*positions))
            assert all(s == score for s in score_group)
            yield cls(region.chrom, positions[0], positions[-1] + 1, score)


def bed_record_from_line(line: str):
    parts = line.split("\t")
    return infer_bed_record_class_from_parts(parts)(*parts)


def infer_bed_record_class_from_parts(parts: List[str]):
    if len(parts) == 3:
        return Bed3Record
    if len(parts) == 4:
        return Bed4Record
    if len(parts) == 5:
        return Bed5Record
    elif len(parts) == 6:
        return Bed6Record
    elif len(parts) == 10:
        try:
            NarrowPeakRecord.from_line_parts(parts)
            return NarrowPeakRecord
        except:
            raise ValueError(f"{parts} is not a valid bed")
    elif len(parts) == 12:
        return Bed12Record
    else:
        raise ValueError(f"{parts} is not a valid bed")


class RegionReader(abc.ABC):
    extensions = None

    def __init__(self, in_file, check_extension=True, record_class=None):
        if "://" not in in_file:
            assert os.path.exists(in_file), f"{in_file} does not exist"
        if check_extension:
            type(self).validate_extension(in_file)

        self.record_class = record_class

        self.in_file = in_file
        self._init_reader(self.in_file)

        # set default record class if necessary
        if record_class is None:
            self.record_class = self.get_default_record_class()
        else:
            self.record_class = record_class

    def _init_reader(self, in_file):
        """
        Initialize the object which will read `in_file`

        :param in_file: path to the input
        """
        self.in_file = in_file

    def __next__(self):
        """Get the next record in the file"""
        return self.record_from_parts(next(self._reader).rstrip("\n").split("\t"))

    @classmethod
    def validate_extension(cls, path):
        assert cls.extensions is not None
        if not any(path.endswith(e) for e in cls.extensions):
            raise ValueError(f"extension for {path} is not one of {cls.extensions}")

    @abc.abstractmethod
    def get_default_record_class(self):
        """
        Gets the default record class to parse each line with.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._reader.close()

    def __iter__(self):
        return self

    @classmethod
    def is_empty(cls, path, check_extension=True):
        with cls(path, check_extension=check_extension) as reader:
            try:
                next(iter(reader))
            except StopIteration:
                return True
            return False

    @classmethod
    def get_preferred_extension(cls):
        return cls.extensions[0]

    @classmethod
    def get_writer_class(cls):
        writer_class_name = cls.__name__.split(".")[-1].replace("Reader", "Writer")
        module = importlib.import_module(f"fbio.formats")
        return getattr(module, writer_class_name)

    def record_from_parts(self, parts):
        return self.record_class.from_line_parts(parts)


class BedReader(RegionReader):
    extensions = ["bed", "bed.gz", "narrowPeak", "narrowPeak.gz", "nPk", "nPk.gz", "tsv", "tsv.gz"]

    def get_default_record_class(self):
        return type(self).infer_bed_record_class_from_file(self.in_file)

    @staticmethod
    def infer_bed_record_class_from_file(in_file):
        with open(in_file) as reader:
            try:
                line = next(reader).rstrip("\n")
                return infer_bed_record_class_from_parts(line.split("\t"))
            except StopIteration:
                return None

    def _init_reader(self, in_file):
        self._reader = open(in_file)

    @classmethod
    def load_dataframe(cls, in_file, *args, **kwargs):
        with open(in_file) as infile:
            first_line_fields = infile.readline().split()
            if first_line_fields[0].lower() in ["chromosome", "chrom", "chr", "contig"]:
                has_header = True
            else:
                has_header = False
        with cls(in_file) as reader:
            record_class = reader.get_default_record_class()
            df = pandas.read_table(
                in_file,
                header=0 if has_header else None,
                names=[f.name for f in fields(record_class)],
                *args,
                **kwargs,
            )
            return df


class IndexedRegionReader(RegionReader):
    @abc.abstractmethod
    def _fetch(self, chrom, start, stop) -> Iterable:
        """Fetch a genomic region from the file, yields an iterable of raw column data, ex ['chr1', '1', '200']"""
        yield

    @abc.abstractmethod
    def _fetch_all(self) -> Iterable:
        """Fetch all the regions from the file, yields an iterable of raw column data, ex ['chr1', '1', '200']"""
        yield

    def fetch(self, chrom=None, start=None, stop=None) -> Iterable[RegionRecord]:
        IndexedRegionReader.validate_fetch_params(chrom, start, stop)
        if chrom is None:
            return map(self.record_from_parts, self._fetch_all())
        else:
            return map(self.record_from_parts, self._fetch(chrom, start, stop))

    @staticmethod
    def validate_fetch_params(chrom, start, stop):
        if chrom is None:
            if not (start is None and stop is None):
                raise ValueError("if chrom is None, start and stop must be None")

    def __iter__(self):
        return self.fetch()

    def __next__(self):
        raise AttributeError("indexed readers do not support next()")

    @classmethod
    def slice(
        cls,
        in_file,
        chrom,
        start,
        stop,
        out_file,
        writer_class=None,
        writer_init_kwargs=None,
        cache_dir=DEFAULT_FBIO_CACHE_DIR,
    ):
        """
        Write all records in `in_file` which overlap `region` and save to `out_file`

        :param in_file: input file
        :param region: region to slice
        :param out_file: output file
        :param cache_dir: cache directory, `out_file` will be a symlink to the cached slice
        :param writer_class: writer class to use
        :param writer_init_kwargs: default writer init kwargs, otherwise will use the reader's header if it exists
        :return:
        """
        if writer_class is None:
            writer_class = cls.get_writer_class()

        log.debug(f"slice(**{locals()})")
        if cache_dir is None:
            return cls._slice(in_file, chrom, start, stop, out_file, writer_class, writer_init_kwargs)
        else:
            """
            1) Get the file hash
            2) See if it exists in the cache_dir
            3) Make the slice if it doesn't exist
            4) Symlink `out_file` to the cached slice
            """
            cache_dir = os.path.expanduser(cache_dir)
            assert os.path.exists(cache_dir), f"{cache_dir} does not exist"
            hash = (
                _get_file_path_and_file_hash_fast(in_file, url_ok=True)
                + f"_{os.path.basename(in_file)}_{chrom}:{start}-{stop}"
            )

            # get the longest matching file extension, use that for the cache path
            possible_extensions = list(filter(lambda ext: out_file.endswith(ext), writer_class.extensions))
            assert len(possible_extensions) > 0
            slice_ext = sorted(possible_extensions, key=len)[-1]
            cache_path = os.path.join(cache_dir, f"{hash}.{slice_ext}")

            # If the cache_path already exists, then use it
            if not os.path.exists(cache_path):
                temp_sub_directory = "".join(random.choice(string.ascii_lowercase) for i in range(30))
                log.debug(f"Creating slice at {cache_path}")

                # create a temp directory to store the data. We will then use an atomic move to avoid the race condition
                # usually we would just create a file, but the slice function can also create an index which wouldn't be
                # moved. Our solution is to create a temp directory and then move everything in the directory.
                temp_dir = os.path.join(cache_dir, temp_sub_directory)
                os.mkdir(temp_dir)

                temp_filename = os.path.join(temp_dir, f"{hash}.{slice_ext}")

                cls._slice(in_file, chrom, start, stop, temp_filename, writer_class, writer_init_kwargs)

                # now move all of the files in temp_dir to the cache directory
                for fname in os.listdir(temp_dir):
                    os.rename(os.path.join(temp_dir, fname), os.path.join(cache_dir, fname))
                # and remove the temp directory
                os.rmdir(temp_dir)

            # remove symlink to slice if it does exist
            if os.path.exists(out_file):
                log.debug("Removing symlink to slice")
                os.unlink(out_file)

            # create symlink to slice
            log.debug("Creating symlink to slice")
            os.symlink(cache_path, out_file)

            if issubclass(writer_class, TabixBedWriter):
                # symlink the tbi as well
                linked_path = os.readlink(out_file)
                source = linked_path + ".tbi"
                target = out_file + ".tbi"
                if os.path.exists(target):
                    os.unlink(target)
                os.symlink(source, target)

    @classmethod
    def _slice(cls, in_file, chrom, start, stop, out_file, writer_class=None, writer_init_kwargs=None):
        log.debug(f"_slice(**{locals()})")
        if writer_class is None:
            writer_class = cls.get_writer_class()

        with cls(in_file) as reader:
            if writer_init_kwargs is None:
                if hasattr(reader, "header"):
                    writer_init_kwargs = dict(header=reader.header)
                else:
                    writer_init_kwargs = dict()

            with writer_class(out_file, **writer_init_kwargs) as writer:
                for rec in reader.fetch(chrom, start, stop):
                    writer.write_record(rec)


class BigWigReader(IndexedRegionReader):
    extensions = ["bigWig", "bw", "bigwig"]

    def _init_reader(self, in_file):
        self._reader = pyBigWig.open(in_file)

    def _fetch_all(self) -> Iterable:
        chroms = list(self._reader.chroms().keys())
        for chrom in chroms:
            for parts in self._fetch(chrom, None, None):
                yield parts

    def _fetch(self, chrom, start=None, stop=None) -> Iterable:
        if start is None:
            start = 0
        if stop is None:
            stop = self._reader.chroms()[chrom]
        for interval in self._reader.intervals(chrom, start, stop) or []:
            if isinstance(interval[2], str):
                parts = interval[2].split("\t")
            else:
                parts = [interval[2]]
            yield [chrom, interval[0], interval[1]] + parts

    def values(self, chrom, start=None, stop=None, allow_invalid_chroms=False):
        """
        Returns values in bigwig in the corresponding region
        :param allow_invalid_chroms: Flag to allow invalid chromosomes to return all 0s instead of raising an error.
         The bedtools bigwig reader will raise an error if passed a chromosome that is not in the corresponding
         reference. However, some bigwig files in public data have non-standard chromosomes that are not found in
         our reference.
        :return: values of a bigwig file in the designated region
        """
        if allow_invalid_chroms:
            if chrom in self._reader.chroms():
                # If the chromosome is not in the bigwig file, we get an "Invalid interval bounds!" RuntimeError
                vals = numpy.array(self._reader.values(chrom, start, stop))
                vals[numpy.isnan(vals)] = 0.0
            else:
                vals = numpy.zeros(stop - start)
        else:
            vals = numpy.array(self._reader.values(chrom, start, stop))
            vals[numpy.isnan(vals)] = 0.0
        return vals

    def stats(self, chrom, start=None, stop=None, nBins=1, exact=False, type="mean"):
        return self._reader.stats(chrom, start, stop, nBins=nBins, exact=exact, type=type)

    @property
    def header(self):
        """kwargs use to initialize a writer that would have the same header as this reader"""
        x = dict(chrom_lengths=dict(self._reader.chroms()))
        return x

    def get_default_record_class(self):
        return WigRecord

    @classmethod
    def _slice(cls, in_file, chrom, start, stop, out_file, writer_class=None, writer_init_kwargs=None):
        if writer_class == TabixBedWriter:
            # use the command line tool for speed.  It's also more reliable than pyBigWig for remote files
            # grep removes header lines like: "#bedGraph section chr1:0-630379"
            _bash(
                f"bigWigToWig -chrom={chrom} -start={start} -end={stop} {in_file} /dev/stdout "
                f'|grep -v -P "^#" | bgzip -c > {out_file}'
            )
            _bash(f"tabix -p bed -f {out_file}")
        else:
            super()._slice(in_file, chrom, start, stop, out_file)


class BigBedReader(BedReader, IndexedRegionReader):
    extensions = ["bb", "bigbed", "bigBed"]

    def _init_reader(self, in_file):
        self.in_file = in_file

    @staticmethod
    def _get_chroms_and_lens(in_file):
        res = subprocess.run(["bigBedInfo", "-chroms", in_file], check=True, stdout=subprocess.PIPE)
        # exctract the chromosome sizes
        lines_iter = iter(io.StringIO(res.stdout.decode()))
        line = next(lines_iter)
        while not line.startswith("chromCount:"):
            line = next(lines_iter)
        assert line.startswith("chromCount:")
        # extract the number of chromosomes
        num_chroms = int(line.strip().split()[1])
        chrom_lengths = {}
        for i in range(num_chroms):
            data = next(lines_iter).strip().split()
            chrom_lengths[data[0]] = int(data[2])
        # make sure we've made it through the contigs
        assert next(lines_iter).startswith("basesCovered")
        return chrom_lengths

    @classmethod
    def is_empty(cls, fname, check_extension=True):
        return len(cls._get_chroms_and_lens(fname)) == 0

    def __init__(self, *args, **kwargs):
        BedReader.__init__(self, *args, **kwargs)
        self.chroms = self._get_chroms_and_lens(self.in_file)
        self._is_empty = len(self.chroms) == 0

    @staticmethod
    def _ucsc_bigbed_fetch_intervals(in_file, chrom, start, stop, window_size=10000):
        """
        bigBedToBed loads the entire query into memory behind the scenes which makes reading 1 record from a large file
        very slow.  This divides up the range into windows to speed things up.
        """
        for start_, stop_ in _windowed_range(start, stop, window_size):
            cmd = [
                "bigBedToBed",
                in_file,
                "/dev/stdout",
                f"-chrom={chrom}",
                f"-start={start_}",
                f"-end={stop_}",
            ]
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            for line in io.StringIO(res.stdout.decode()):  ## is this necessary??
                yield line.split()

    def _fetch(self, chrom, start=None, stop=None):
        if self._is_empty:
            return

        # reader is None when the file has no records
        if start is None:
            start = 0
        if stop is None:
            stop = self.chroms[chrom]

        for parts in self._ucsc_bigbed_fetch_intervals(self.in_file, chrom, start, stop):
            yield parts

    def _fetch_all(self):
        if not BigBedReader.is_empty(self.in_file):
            for chrom in self.chroms:
                for parts in self._fetch(chrom, None, None):
                    yield parts

    def close(self):
        pass

    @classmethod
    def _slice(cls, in_file, chrom, start, stop, out_file, writer_class=None, writer_init_kwargs=None):
        if writer_class == TabixBedWriter:
            # use the command line tool for speed.  It's also more reliable than pyBigWig for remote files
            _bash(
                f"bigBedToBed -chrom={chrom} -start={start} -end={stop} {in_file} /dev/stdout "
                f"| bgzip -c > {out_file}"
            )
            _bash(f"tabix -p bed -f {out_file}")
        else:
            if cls.is_empty(in_file):
                # we can't read an empty bigbed, so just copy it since we know the slice will be empty
                shutil.copy(in_file, out_file)
            else:
                super()._slice(in_file, chrom, start, stop, out_file, writer_class, writer_init_kwargs)

    @property
    def header(self):
        return dict(chrom_lengths=dict(self.chroms))

    @staticmethod
    def infer_bed_record_class_from_file(in_file):
        if BigBedReader.is_empty(in_file):
            return Bed3Record
        else:
            parts = BigBedReader.read_first_line(in_file)
            return infer_bed_record_class_from_parts(parts)

    @staticmethod
    def read_first_line(in_file):
        # TODO -- if we just care about the parts we can use bigBedInfo
        cmd = f"bigBedToBed -maxItems=1 {in_file} /dev/stdout"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        rv = io.StringIO(res.stdout.decode()).readline()
        if rv == "":
            raise StopIteration("No line to read")
        else:
            return rv.split()


class TabixBedReader(IndexedRegionReader, BedReader):
    extensions = ["bed.gz", "tsv.gz"]

    def _init_reader(self, in_file):
        if not _is_s3_uri(in_file):
            # need the absolute path, since we set the cwd to a temp directory to download any index files
            in_file = os.path.abspath(in_file)

        super()._init_reader(in_file)
        # Tabix will download an index to the local directory if it is an s3 url, so we use a temp_dir as the cwd
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chroms = (
            subprocess.check_output(["tabix", "-l", in_file], cwd=self.temp_dir.name)
            .decode()
            .strip()
            .split("\n")
        )

    def _fetch(self, chrom, start, stop) -> Iterable:
        def optional(x):
            return None if x is None else x

        """Fetch a genomic region from the file, yields an iterable of raw column data, ex ['chr1', '1', '200']"""
        region_str = region_to_region_str(chrom, optional(start), optional(stop))
        cmd = ["tabix", os.path.abspath(self.in_file), region_str]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=self.temp_dir.name)
        for line in proc.stdout:
            line = line.decode().rstrip()
            if line != "":
                parts = line.split("\t")
                # tabix is 1 based inclusive!!! check that this region actually intersects
                start2, stop2 = int(parts[1]), int(parts[2])
                if intervals_intersect_with_none(start, stop, start2, stop2):
                    yield parts
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(returncode=proc.returncode, cmd=f'{" ".join(cmd)}')

    def _fetch_all(self) -> Iterable:
        with smart_open.open(self.in_file) as reader:
            for line in reader:
                yield line.rstrip("\n").split("\t")

    def get_default_record_class(self):
        """
        Gets the default record class to parse each line with.
        """
        try:
            parts = next(iter(self._fetch_all()))
            return infer_bed_record_class_from_parts(parts)
        except StopIteration:
            return None

    def count_entries_overlapping_regions(self, regions):
        """Return a count of entries that overlap each region."""
        rv = []
        for region in regions:
            rv.append(sum(1 for _ in self.fetch(region.chrom, region.start, region.stop)))
        return rv

    def close(self):
        self.temp_dir.cleanup()
        pass

    @classmethod
    def can_open(cls, in_file):
        # flake8: noqa: E722
        try:
            with cls(in_file):
                return True
        except:
            return False


class FragmentBedReader(TabixBedReader):
    extensions = ["bed.gz"]

    def get_default_record_class(self):
        return Fragment


class MethylFragmentBedReader(TabixBedReader):
    extensions = ["methyl.bed.gz"]

    def get_default_record_class(self):
        return Fragment

    def record_from_parts(self, parts):
        return self.record_class.from_bed12_parts(parts)


class FragmentBigBedReader(BigBedReader):
    def get_default_record_class(self):
        return Fragment


class NarrowPeakValueError(ValueError):
    pass


@dataclass
class NarrowPeakRecord(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    score: int
    strand: str
    signal_value: float
    p_value: float
    q_value: float
    peak: int
    strict: InitVar[bool] = False

    def __post_init__(self, strict):
        super().__post_init__(strict=strict)
        if self.peak is not None:
            if self.peak < 0 or (self.peak > (self.stop - self.start)):
                raise NarrowPeakValueError(f"Peak is outside bounds of the region: {self}")


class NarrowPeakBedReader(TabixBedReader):
    extensions = [
        "narrowPeak",
        "narrowPeak.gz",
        "narrowPeak.bed",
        "narrowPeak.bed.gz",
        "nPk",
        "nPk.gz",
        "nPk.bed",
        "nPk.bed.gz",
    ]

    def get_default_record_class(self):
        return NarrowPeakRecord


class BedGraphReader:
    def __init__(self, bedgraph_file: Optional[str] = None):
        if bedgraph_file is not None:
            self.bedgraph_file = bedgraph_file
            df = pandas.read_csv(
                self.bedgraph_file,
                sep="\t",
                skiprows=1,
                names=["chrom", "start", "stop", "value"],
                dtype={"chrom": str, "start": int, "stop": int, "value": float},
            ).dropna()

            self.data = {}
            for chrom, group in df.groupby("chrom"):
                self.data[chrom] = numpy.array(group[["start", "stop", "value"]])

    def get_idxs_for_region(self, chrom, start, stop, verbose=True):
        """
        Finds the start and stop indexes that bound the desired region
        """
        chrom_data = self.data[chrom]

        start_idx = bisect.bisect_left(chrom_data[:, 0], start)

        if verbose and start_idx == 0 and start < chrom_data[start_idx][0]:
            warnings.warn(
                f"Region {chrom}:{start}-{stop} is outside the bounds, returning interior indexes only"
            )

        # We expect the stop to be somewhat close to the start, so we start our search there
        # This loop gives us exponential growth of the upper bound
        ii = 0
        stop_idx = start_idx + ii
        while stop_idx < len(chrom_data) and chrom_data[stop_idx][1] < stop:
            stop_idx = min(start_idx + 2 ** ii, len(chrom_data))
            ii += 1

        stop_idx = bisect.bisect_right(chrom_data[:, 1], stop, lo=start_idx, hi=stop_idx)

        return start_idx, stop_idx

    def fetch(self, chrom, start, stop, verbose=True):
        start_idx, stop_idx = self.get_idxs_for_region(chrom, start, stop, verbose=verbose)
        assert stop_idx >= start_idx, f"Found stop {stop_idx} less than start {start_idx}"
        assert stop_idx <= len(self.data[chrom])

        return self.data[chrom][start_idx:stop_idx]


class MethylBedGraphReader(BedGraphReader):
    def __init__(self, bedgraph_file: str, check_format=True):
        super().__init__(None)
        if check_format:
            assert bedgraph_file.endswith("_CpG.bedGraph"), "Expecting MethylDackel file"
        self.bedgraph_file = bedgraph_file
        m_df = pandas.read_csv(
            self.bedgraph_file,
            sep="\t",
            skiprows=1,
            names=["chrom", "start", "stop", "pct_methyl", "methylated", "unmethylated"],
            dtype={
                "chrom": str,
                "start": int,
                "stop": int,
                "pct_methyl": float,
                "methylated": int,
                "unmethylated": int,
            },
        ).dropna()

        self.data = {}
        for chrom, group in m_df.groupby("chrom"):
            self.data[chrom] = numpy.array(
                group[["start", "stop", "pct_methyl", "methylated", "unmethylated"]]
            )

    def fetch(self, chrom, start, stop, verbose=True):
        """
        Returns the total number of methylated and unmethylated CpG-reads in a region
        """
        start_idx, stop_idx = self.get_idxs_for_region(chrom, start, stop, verbose=verbose)
        assert stop_idx >= start_idx, f"Found stop {stop_idx} less than start {start_idx}"
        assert stop_idx <= len(self.data[chrom])

        num_meth, num_unmeth = 0, 0
        for pos_meth_data in self.data[chrom][start_idx : stop_idx + 1]:
            if start <= pos_meth_data[0] < stop:
                num_meth += pos_meth_data[3]
                num_unmeth += pos_meth_data[4]

        return num_meth, num_unmeth


class NarrowPeakBigBedReader(BigBedReader):
    extensions = ["narrowPeak.bb", "nPk.bb"]

    def get_default_record_class(self):
        return NarrowPeakRecord


class RegionWriter(abc.ABC):
    extensions = None

    def __init__(self, out_file, check_extension=True):
        if check_extension and self.extensions is not None:
            if not any(out_file.endswith(e) for e in self.extensions):
                raise ValueError(f"extension for {out_file} is not one of {self.extensions}")

        file_dir = os.path.dirname(os.path.abspath(out_file))
        if not os.access(file_dir, os.W_OK):
            raise PermissionError(f"No permission to write {file_dir}, os.stat: {os.stat(file_dir)}")

        self.out_file = out_file
        self._init_writer()

    @classmethod
    def get_preferred_extension(cls):
        return cls.extensions[0]

    def _init_writer(self):
        self._writer = smart_open(self.out_file)

    def __enter__(self):
        return self

    def close(self):
        self._writer.close()

    def __exit__(self, *args):
        self.close()

    @abc.abstractmethod
    def write_record(self, record):
        """Write a record to the file"""
        pass

    def write_records(self, records):
        for record in records:
            self.write_record(record)

    @classmethod
    def get_reader_class(cls):
        reader_class_name = cls.__name__.split(".")[-1].replace("Writer", "Reader")
        module = inspect.getmodule(cls)
        return getattr(module, reader_class_name)


class BedWriter(RegionWriter):
    extensions = ["bed", "bed.gz", "narrowPeak", "narrowPeak.gz", "nPk", "nPk.gz", "tsv", "tsv.gz"]

    def write_record(self, record: RegionRecord):
        self._writer.write((record.to_line() + "\n").encode())


class TabixBedWriter(BedWriter):
    extensions = ["bed.gz", "tsv.gz"]

    def close(self):
        super().close()
        pysam.tabix_index(self.out_file, preset="bed", force=True)


class FragmentBedWriter(TabixBedWriter):
    extensions = ["bed.gz"]

    def write_record(self, record: Fragment):
        self._writer.write((record.to_line() + "\n").encode())


class BigWigWriter(RegionWriter):
    """pyBigWig doesnt have any bigbed specific writing features"""

    extensions = ["bigWig", "bw", "bigwig"]

    def __init__(self, out_file, header: dict, check_extension=True):
        warnings.warn(
            f"For some reason, IGV will not properly vizualize bigwigs created with this class.  "
            f"Consider creating a bedgraph and then using bedGraphToBigWig"
        )

        if "chrom_lengths" not in header:
            raise KeyError(f"chrom_lengths is a required header key")

        self.header = header
        super().__init__(out_file, check_extension=check_extension)

    def _init_writer(self):
        self._writer = pyBigWig.open(self.out_file, "wb")
        self._writer.addHeader(list(self.header["chrom_lengths"].items()))

    def write_record(self, record: WigRecord):
        assert record.chrom in self.header["chrom_lengths"], (
            f"{record.chrom} is invalid, " f'must be one of {self.header["chrom_lengths"].keys()}'
        )
        if not isinstance(record.score, float):
            raise TypeError(f"pybigwig can only write floats, record.score is {record.score}")
        self._writer.addEntries([record.chrom], [record.start], ends=[record.stop], values=[record.score])


class BigBedWriter(BedWriter):
    """
    There are no python libraries for writing bigbeds
    Our hack is to write beds, then convert to bigbed on the using bedToBigBed at close()
    """

    extensions = ["bb", "bigbed", "bigBed"]

    def __init__(self, out_file, header: dict, check_extension=True):
        if "chrom_lengths" not in header:
            raise KeyError(f"chrom_lengths is a required header key")
        self.header = header
        super().__init__(out_file, check_extension=check_extension)

        # TODO optionalgzipwriter is writing ungzipped bed files right now..

    def write_record(self, record: RegionRecord):
        if record.chrom not in self.header["chrom_lengths"]:
            raise ValueError(f"{record.chrom} is not a valid chrom for this bigbed")

        super().write_record(record)

    def close(self):
        super().close()
        with TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, os.path.basename(self.out_file))
            shutil.move(self.out_file, temp_path)
            with open("chrom.sizes", "w") as fp:
                for chrom, length in self.header["chrom_lengths"].items():
                    fp.write(f"{chrom}\t{length}\n")

            if BedReader.is_empty(temp_path, check_extension=False):
                # default bed type
                params = ["-type=bed4"]
            else:
                params = []

            # warning: bedToBigBed is picky about the output, and likely to fail on public data
            # consider writing using a different format
            subprocess.run(["bedToBigBed"] + params + [temp_path, "chrom.sizes", self.out_file], check=True)


class NarrowPeakBedWriter(TabixBedWriter):
    extensions = ["narrowPeak.bed.gz", ".bed.gz", "narrowPeak.gz"]


class NarrowPeakBigBedWriter(BigBedWriter):
    extensions = ["narrowPeak.bb", ".bb"]


@dataclass
class GenePredRecord(RegionRecord):
    chrom: str
    start: int
    stop: int
    name: str
    score: int
    strand: str
    thickStart: int
    thickEnd: int
    itemRgb: float
    block_count: int

    exon_start: str
    exon_end: str
    strict: InitVar[bool] = False


class GenePredWriter(TabixBedWriter):
    pass


class GenePredReader(TabixBedReader):
    def get_default_record_class(self):
        return GenePredRecord


def get_readers_which_support_path(file_path, only_index_readers=False):
    r = []
    for cls in _get_subclasses_of(RegionReader):
        if only_index_readers and not issubclass(cls, IndexedRegionReader):
            continue
        if any(file_path.endswith("." + ext) for ext in (cls.extensions or [])):
            r.append(cls)
    return r


def get_default_reader_class(file_path):
    def is_match(extensions):
        return any(file_path.endswith("." + ext) for ext in extensions)

    # We start with the most specific case and get broader
    if is_match(NarrowPeakBedReader.extensions):
        return NarrowPeakBedReader
    elif is_match(TabixBedReader.extensions) and TabixBedReader.can_open(file_path):
        return TabixBedReader
    elif is_match(BedReader.extensions):
        return BedReader
    elif is_match(NarrowPeakBigBedReader.extensions):
        return NarrowPeakBigBedReader
    elif is_match(BigBedReader.extensions):
        return BigBedReader
    elif is_match(BigWigReader.extensions):
        return BigWigReader
    else:
        raise ValueError(f"Cannot get default reader for {file_path}")
