from pyliftover import LiftOver


class RegionLiftOver(LiftOver):
    """A liftover class that allows for region conversions."""

    def __init__(self, source, dest, *args, **kwargs):
        valid_assemblies = {"hg16", "hg17", "hg18", "hg19", "hg38"}
        assert source in valid_assemblies, f"invalid source: {source}, should be {valid_assemblies}"
        assert dest in valid_assemblies, f"invalid dest: {dest}, should be {valid_assemblies}"
        self.source = source
        self.dest = dest
        super().__init__(source, dest, *args, **kwargs)

    def uniquely_convert_coordinate(self, chrom, coordinate, strand=None):
        """Return new coordinate if coordinate is uniquely convertible, else None."""
        converted = self.convert_coordinate(chrom, coordinate, strand=strand)
        if converted is None or len(converted) != 1:
            return None
        return converted.pop()

    def uniquely_convert_region(self, chrom, start, stop, strand=None):
        if start >= stop:
            raise ValueError("Stop coordinate must be greater than the start coordinate.")
        converted_start = self.uniquely_convert_coordinate(chrom, start, strand)
        if converted_start is None:
            return None
        converted_stop = self.uniquely_convert_coordinate(chrom, stop, strand)
        if converted_stop is None:
            return None
        # check that they have the same chromosome
        if converted_start[0] != converted_stop[0]:
            return None

        # unpack everything
        new_chrom = converted_start[0]
        # it's unlikely for the region to have the opposite strand in the
        # new coordinates, but it's good to check regardless
        if strand is not None:
            new_strand = converted_start[2]
        else:
            new_strand = None
        if converted_start[2] != converted_stop[2]:
            return None

        if converted_start[1] >= converted_stop[1]:
            return None

        new_start = converted_start[1]
        new_stop = converted_stop[1]

        # make sure the interval length is the same
        if (new_stop - new_start) != (stop - start):
            return None

        return new_chrom, new_start, new_stop, new_strand
