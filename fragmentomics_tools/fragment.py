import numpy
from dataclasses import fields, dataclass, InitVar
from typing import Iterable, List, Union, Optional, TextIO

class ClassPropertyDescriptor(object):
    """
    used by classproperty() to make @classproperty decorator
    """

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """
    @classproperty decorator
    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)

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
