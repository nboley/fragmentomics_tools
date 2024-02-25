import os
import tempfile
from dataclasses import dataclass, InitVar

import pytest
from pytest import mark, raises

# TODO -- move this into config

def get_test_path(*args):
    return os.path.join("/home/nboley/src/fragmentomics_tools/test/data", *args)

DEFAULT_CACHE_DIR = get_test_path("./")

from fragmentomics_tools.formats import (
    RegionRecord,
    BedReader,
    TabixBedWriter,
    FragmentBedReader,
    FragmentBedWriter,
    bed_record_from_line,
    Bed6Record,
    GappedPeakRecord,
    IndexedRegionReader,
    BigWigWriter,
    BigBedWriter,
    WigRecord,
    TabixBedReader,
    BigWigReader,
    BedWriter,
    get_default_reader_class,
    BigBedReader,
    Bed4Record,
    Bed3Record,
    BedIntervalTreeReader,
)
from fragmentomics_tools.fragment import Fragment
from fragmentomics_tools.region import Region

def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

FRAG_BED_GZ = get_test_path("frags.bed.gz")

BED_RECS = [
    Bed6Record("chr1", 1, 200, ".", 1.0, "+"),
    Bed6Record("chr1", 200, 300, ".", 1.0, "+"),
    Bed6Record("chr1", 300, 400, ".", 1.0, "+"),
    Bed6Record("chr2", 100, 200, ".", 1.0, "+"),
]
BED4_RECS = [Bed4Record(rec.chrom, rec.start, rec.stop, "asdf") for rec in BED_RECS]
FRAG_RECS = [Fragment(rec.chrom, rec.start, rec.stop, 0, 1, 0.5) for rec in BED_RECS]
WIGGLE_RECS = [WigRecord(rec.chrom, rec.start, rec.stop, i + 0.5) for i, rec in enumerate(BED_RECS)]

READER_CLASSES = [
    BedReader,
    TabixBedReader,
    FragmentBedReader,
    BigWigReader,
    BigBedReader,
    BedIntervalTreeReader,
]
WRITER_CLASSES = [cls.get_writer_class() for cls in READER_CLASSES]


@pytest.mark.parametrize("writer_class", WRITER_CLASSES)
def test_extensions_match(writer_class):
    assert writer_class.extensions == writer_class.get_reader_class().extensions


def test_custom_record(cleandir):
    @dataclass
    class MyBedRecord(RegionRecord):
        chrom: str
        start: int
        stop: int
        extra_field: str
        strict: InitVar[bool] = False

    with TabixBedWriter("test.bed.gz") as writer:
        writer.write_records(BED4_RECS)

    with TabixBedReader("test.bed.gz", record_class=MyBedRecord) as reader:
        recs = list(reader)
        assert all(isinstance(r, MyBedRecord) for r in recs)


def test_tabix(cleandir):
    with TabixBedWriter("test.bed.gz") as writer:
        for rec in BED_RECS:
            writer.write_record(rec)

    with TabixBedReader("test.bed.gz") as reader:
        assert list(reader) == BED_RECS


def test_get_default_reader_for_file(cleandir):
    assert get_default_reader_class("test.bed.gz") == BedReader
    assert get_default_reader_class("test.bb") == BigBedReader

    with TabixBedWriter("tabixed.bed.gz") as writer:
        writer.write_record(BED_RECS[0])

    assert get_default_reader_class("tabixed.bed.gz") == TabixBedReader


def get_generic_reader_writer_params(indexed_reader_writers_only=False):
    for reader_class in READER_CLASSES:
        writer_class = reader_class.get_writer_class()
        if indexed_reader_writers_only and not issubclass(reader_class, IndexedRegionReader):
            continue

        for extension in writer_class.extensions:
            if writer_class is FragmentBedWriter:
                records = FRAG_RECS
            elif writer_class is BigWigWriter:
                records = WIGGLE_RECS
            elif writer_class is BigBedWriter:
                records = BED4_RECS
            else:
                records = BED_RECS

            if writer_class in [BigWigWriter, BigBedWriter]:
                writer_init_kwargs = dict(header=dict(chrom_lengths=dict([("chr1", 10000), ("chr2", 10000)])))
            else:
                writer_init_kwargs = dict()

            yield reader_class, writer_class, extension, records, writer_init_kwargs


@mark.parametrize(
    "reader_class, writer_class, extension, records, writer_init_kwargs",
    list(get_generic_reader_writer_params()),
)
def test_generic_region_reader_writer(
    cleandir, reader_class, writer_class, extension, records, writer_init_kwargs
):
    """
    Test non-index specific reading/writing behavior on both indexed and non indexed Reader/Writers
    """
    fname = f"x.{extension}"
    with writer_class(fname, **writer_init_kwargs) as writer:
        for rec in records:
            writer.write_record(rec)

    # FIXME: Why do we have this assertion/? If reader_class == BedReader, file is .narrowPeak,
    #  get_default_reader_class should be NarrowPeakBedReader, which is subclass of BedReader
    #  .
    #  On the other hand, if reader_class == BedIntervalTreeReader, we only get BedReader classes,
    #  so reader_class is a subset of default_reader_class

    # assert issubclass(reader_class, get_default_reader_class(fname))
    # assert issubclass(get_default_reader_class(fname), reader_class)
    assert not reader_class.is_empty(fname)

    with reader_class(fname) as reader:
        reader.get_default_record_class()
        read_records = list(reader)
        assert read_records == records

        # check functions that require an index
        if not issubclass(reader_class, IndexedRegionReader):
            with pytest.raises(AttributeError):
                reader.fetch()

    # test empty file
    os.unlink(fname)
    with writer_class(fname, **writer_init_kwargs):
        pass
    assert reader_class.is_empty(fname)

    with reader_class(fname) as reader:
        assert list(reader) == []


@mark.parametrize(
    "reader_class, writer_class, extension, records, writer_init_kwargs",
    list(get_generic_reader_writer_params(indexed_reader_writers_only=True)),
)
def test_indexed_region_reader_writer(
    cleandir, reader_class, writer_class, extension, records, writer_init_kwargs
):
    """
    Test the index-specific behaviors of readers/writers
    """
    fname = f"x.{extension}"
    with writer_class(fname, **writer_init_kwargs) as writer:
        for rec in records:
            writer.write_record(rec)

    with reader_class(fname) as reader:
        recs = list(reader.fetch("chr1"))
        assert len(recs) == 3
        assert all(rec.chrom == "chr1" for rec in recs)

        assert list(reader) == records

        assert list(reader.fetch()) == records

        # check bad fetch parameters raises an error
        with pytest.raises(ValueError):
            reader.fetch(None, 1)
            reader.fetch(None, None, 2)

    # test slicing the file we just made
    region = Region("chr1", 200, 301)
    sliced_fname = f"sliced.{extension}"
    reader_class.slice(fname, region.chrom, region.start, region.stop, sliced_fname, cache_dir=None)
    with reader_class(sliced_fname) as slice_reader:
        sliced_records = list(slice_reader)
        assert sliced_records == records[1:3]
        assert not os.path.islink(sliced_fname)

    # test off by ones
    for region, expected in [
        (Region("chr1", 200, 301), records[1:3]),
        (Region("chr1", 200, 300), records[1:2]),
        (Region("chr1", 201, 299), records[1:2]),
        (Region("chr1", 200, 301), records[1:3]),
        (Region("chr1", 100, 200), records[:1]),
        (Region("chr1", 100, 201), records[:2]),
    ]:
        with reader_class(fname) as reader:
            res = list(reader.fetch(region.chrom, region.start, region.stop))
            assert res == expected


@mark.parametrize(
    "reader_class, writer_class, extension, records, writer_init_kwargs",
    list(get_generic_reader_writer_params(indexed_reader_writers_only=True)),
)
def test_cached_slicing(cleandir, reader_class, writer_class, extension, records, writer_init_kwargs):
    fname = f"x.{extension}"
    with writer_class(fname, **writer_init_kwargs) as writer:
        for rec in records:
            writer.write_record(rec)

    region = Region("chr1", 200, 301)

    def slice_from_cache_get_mtime(out_path):
        reader_class.slice(
            fname, region.chrom, region.start, region.stop, out_path, cache_dir="cache_dir",
        )
        with reader_class(out_path) as slice_reader:
            sliced_records = list(slice_reader)
            assert sliced_records == records[1:3]
            # make sure this is a symlink
            assert os.path.islink(out_path)
            mtime = os.path.getmtime(out_path)
            return mtime

    mkdir_p("cache_dir")

    mtime1 = slice_from_cache_get_mtime(f"slice_with_cache.{extension}")
    mtime2 = slice_from_cache_get_mtime(f"slice_with_cache.2.{extension}")
    mtime3 = slice_from_cache_get_mtime(f"slice_with_cache.3.{extension}")

    assert mtime1 == mtime2, "mtime should not have changed"
    assert mtime1 == mtime3, "mtime3 should not have changed"


def test_fragment_bed_mapq_only_reader():
    with FragmentBedReader(get_test_path("frags_mapq_only.bed.gz")) as fbr:
        frags = list(fbr)
        assert frags[0].mapq1 == 27
        assert frags[0].mapq2 is None
        assert frags[0].mapq12_min == 27
        assert frags[1].mapq1 == 0


def test_bed_record():
    bed_line = "chr1\t2\t3"
    bed_rec = Bed3Record(chrom="chr1", start=2, stop=3)
    assert Bed3Record.from_line(bed_line) == bed_rec
    assert bed_record_from_line(bed_line) == bed_rec
    with pytest.raises(TypeError):
        # wrong number of columns
        Bed6Record.from_line(bed_line)
        GappedPeakRecord.from_line(bed_line)

    bed6_line = "chr11\t2\t3\tx\t1.1\t+"
    bed6_rec = Bed6Record(chrom="chr11", start=2, stop=3, name="x", score=1.1, strand="+")
    with pytest.raises(TypeError):
        # wrong number of columns
        Bed3Record.from_line(bed6_line)
        GappedPeakRecord.from_line(bed6_line)

    assert bed_record_from_line(bed6_line) == bed6_rec


def test_bed_record_casting():
    @dataclass
    class BedRec(RegionRecord):
        chrom: str
        start: int
        stop: int
        name: str
        strand: str
        pval: float
        strict: InitVar[bool] = True

    with raises(ValueError):
        # stop coordinate is a float string, not an int
        BedRec("a", 1, "2.2", "c", "+", 1.0)

    with raises(ValueError):
        # pval=1 is an int string, not a float
        assert BedRec("a", 1, 2, ".", ".", pval="1")

    assert BedRec("a", 1, 2, "c", "+", "1", strict=False) == BedRec(
        chrom="a", start=1, stop=2, name="c", strand="+", pval="1", strict=False
    )

    assert BedRec("a", 1, "2", "c", "+", 1.0) == BedRec(
        chrom="a", start=1, stop=2, name="c", strand="+", pval=1.0
    )
    assert BedRec("a", 1, "2", "c", ".", 1.0) == BedRec(
        chrom="a", start=1, stop=2, name="c", strand=".", pval=1.0
    )
    assert BedRec("a", 1, 2, ".", ".", 1.0) == BedRec(
        chrom="a", start=1, stop=2, name=None, strand=".", pval=1.0
    )
    assert BedRec("a", 1, 2, ".", ".", 1.0) == BedRec(
        chrom="a", start=1, stop=2, name=None, strand=".", pval=1.0
    )
    assert BedRec("a", 1, 2, ".", None, 1.0) == BedRec(
        chrom="a", start=1, stop=2, name=".", strand=None, pval=1.0
    )
    assert BedRec("a", 1, 2, None, None, 1.0) == BedRec(
        chrom="a", start=1, stop=2, name=None, strand=None, pval=1.0
    )


def test_slice_encode_big_wig():
    with tempfile.NamedTemporaryFile(suffix=".bw") as test_bw:
        BigWigReader._slice(
            "https://www.encodeproject.org/files/ENCFF367ZVE/@@download/ENCFF367ZVE.bigWig",
            "chr1",
            9999985,
            10000065,
            test_bw.name,
        )
        with BigWigReader(test_bw.name) as reader:
            recs = list(reader.fetch("chr1"))
            assert recs == [
                WigRecord(chrom="chr1", start=9999985, stop=10000005, score=0.0),
                WigRecord(chrom="chr1", start=10000005, stop=10000025, score=0.0),
                WigRecord(chrom="chr1", start=10000025, stop=10000045, score=0.0),
                WigRecord(chrom="chr1", start=10000045, stop=10000065, score=0.0),
            ]


@mark.parametrize("cache_dir", [DEFAULT_CACHE_DIR, None])
@mark.parametrize("reader_class, records", [(BigBedReader, BED4_RECS), (BigWigReader, WIGGLE_RECS)])
def test_slice_with_different_writer(cleandir, cache_dir, reader_class, records):
    writer_class = reader_class.get_writer_class()
    ext = writer_class.extensions[0]
    fname = f"test.{ext}"
    with writer_class(fname, header=dict(chrom_lengths=dict([("chr1", 10000), ("chr2", 10000)]))) as writer:
        writer.write_records(records)
    region = Region("chr1", 200, 302)

    reader_class.slice(
        fname,
        region.chrom,
        region.start,
        region.stop,
        "test.bed.gz",
        TabixBedWriter,
        dict(),
        cache_dir=cache_dir,
    )
    with TabixBedReader("test.bed.gz") as reader:
        recs = list(reader)
        # comparing with to_line() because recs will be Bed4Records, not WiggleRecord
        assert [r.to_line() for r in recs] == [r.to_line() for r in records[1:3]]


def test_is_bed_empty(cleandir):
    # note this has much better coverage in test_generic_reader_writer
    with BedWriter("empty.bed"):
        pass
    assert BedReader.is_empty("empty.bed")

    with BedWriter("not_empty.bed.gz") as writer:
        writer.write_record(Bed3Record("chr1", 1, 2))

    assert not BedReader.is_empty("not_empty.bed.gz")
