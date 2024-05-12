import os

import pandas

from fragmentomics_tools.dataframe import DataFrameBase
BASE_DIR = "/home/nboley/src/fragmentomics_tools/fragmentomics_tools/public_data_resources"

CHROMHMM_STATES = {
    "blueprint": {
        "E1": "Repressed Polycomb High signal H3K27me3",
        "E2": "Repressed Polycomb Low signal H3K27me3",
        "E3": "Low signal",
        "E4": "Heterochromatin High Signal H3K9me3",
        "E5": "Transcription High signal H3K36me3",
        "E6": "Transcription Low signal H3K36me3",
        "E7": "Genic Enhancer High Signal H3K4me1 & H3K36me3",
        "E8": "Enhancer High Signal H3K4me1",
        "E9": "Active Enhancer High Signal H3K4me1 & H3K27Ac",
        "E10": "Distal Active Promoter (2Kb) High Signal H3K4me3 & H3K27Ac & H3K4me1",
        "E11": "Active TSS High Signal H3K4me3 & H3K4me1",
        "E12": "Active TSS High Signal H3K4me3 & H3K27Ac",
    },
    "roadmap_15": {
        "1_TssA": "Active TSS",
        "2_TssAFlnk": "Flanking Active TSS",
        "3_TxFlnk": "Transcr. at gene 5' and 3'",
        "4_Tx": "Strong transcription",
        "5_TxWk": "Weak transcription",
        "6_EnhG": "Genic enhancers",
        "7_Enh": "Enhancers",
        "8_ZNF/Rpts": "ZNF genes & repeats",
        "9_Het": "Heterochromatin",
        "10_TssBiv": "Bivalent/Poised TSS",
        "11_BivFlnk": "Flanking Bivalent TSS/Enh",
        "12_EnhBiv": "Bivalent Enhancer",
        "13_ReprPC": "Repressed PolyComb",
        "14_ReprPCWk": "Weak Repressed PolyComb",
        "15_Quies": "Quiescent/Low",
    },
    "roadmap_18": {
        "1_TssA": "Active TSS",
        "2_TssFlnk": "Flanking TSS",
        "3_TssFlnkU": "Flanking TSS Upstream",
        "4_TssFlnkD": "Flanking TSS Downstream",
        "5_Tx": "Strong transcription",
        "6_TxWk": "Weak transcription",
        "7_EnhG1": "Genic enhancer1",
        "8_EnhG2": "Genic enhancer2",
        "9_EnhA1": "Active Enhancer 1",
        "10_EnhA2": "Active Enhancer 2",
        "11_EnhWk": "Weak Enhancer",
        "12_ZNF/Rpts": "ZNF genes & repeats",
        "13_Het": "Heterochromatin",
        "14_TssBiv": "Bivalent/Poised TSS",
        "15_EnhBiv": "Bivalent Enhancer",
        "16_ReprPC": "Repressed PolyComb",
        "17_ReprPCWk": "Weak Repressed PolyComb",
        "18_Quies": "Quiescent/Low",
    },
}

CHROMHMM_STATES_SHORT = {
    "blueprint": {
        "E1": "PCRepr K27me3",
        "E2": "PCRepr",
        "E3": "Low",
        "E4": "Hetero",
        "E5": "Trans K36me3",
        "E6": "Trans",
        "E7": "Genic Enhancer",
        "E8": "Enhancer K4me1",
        "E9": "Active Enhancer",
        "E10": "Dist Active Prom",
        "E11": "Active TSS K4me1",
        "E12": "Active TSS K27Ac",
    },
}

CHROMHMM_COLORS = {
    "blueprint": {
        "E1": "Silver",
        "E2": "Gainsboro",
        "E3": "White",
        "E4": "PaleTurquoise",
        "E5": "Green",
        "E6": "DarkGreen",
        "E7": "GreenYellow",
        "E8": "Yellow",
        "E9": "Orange",
        "E10": "OrangeRed",
        "E11": "Red",
        "E12": "IndianRed",
    },
    "roadmap_15": {
        "1_TssA": "Red",
        "2_TssAFlnk": "OrangeRed",
        "3_TxFlnk": "LimeGreen",
        "4_Tx": "Green",
        "5_TxWk": "DarkGreen",
        "6_EnhG": "GreenYellow",
        "7_Enh": "Yellow",
        "8_ZNF/Rpts": "MediumAquamarine",
        "9_Het": "PaleTurquoise",
        "10_TssBiv": "IndianRed",
        "11_BivFlnk": "DarkSalmon",
        "12_EnhBiv": "DarkKhaki",
        "13_ReprPC": "Silver",
        "14_ReprPCWk": "Gainsboro",
        "15_Quies": "White",
    },
    "roadmap_18": {
        "1_TssA": "Red",
        "2_TssFlnk": "OrangeRed",
        "3_TssFlnkU": "OrangeRed",
        "4_TssFlnkD": "OrangeRed",
        "5_Tx": "Green",
        "6_TxWk": "DarkGreen",
        "7_EnhG1": "GreenYellow",
        "8_EnhG2": "GreenYellow",
        "9_EnhA1": "Orange",
        "10_EnhA2": "Orange",
        "11_EnhWk": "Yellow",
        "12_ZNF/Rpts": "MediumAquamarine",
        "13_Het": "PaleTurquoise",
        "14_TssBiv": "IndianRed",
        "15_EnhBiv": "DarkKhaki",
        "16_ReprPC": "Silver",
        "17_ReprPCWk": "Gainsboro",
        "18_Quies": "White",
    },
}


# The Roadmap annotations above assume the 'number_mnemonic' schema, but we may have scenarios where we're given
# only one or the other. Add these to each annotation set
assert set(CHROMHMM_STATES.keys()) == set(CHROMHMM_COLORS.keys())

for annot_set in CHROMHMM_STATES.keys():
    _keys = list(CHROMHMM_STATES[annot_set].keys())
    assert set(_keys) == set(CHROMHMM_COLORS[annot_set].keys())
    if annot_set == "blueprint":
        for chromhmm_key in _keys:
            number = chromhmm_key[1:]
            CHROMHMM_STATES[annot_set][number] = CHROMHMM_STATES[annot_set][chromhmm_key]
            CHROMHMM_STATES_SHORT[annot_set][number] = CHROMHMM_STATES_SHORT[annot_set][chromhmm_key]
            CHROMHMM_COLORS[annot_set][number] = CHROMHMM_COLORS[annot_set][chromhmm_key]
    elif annot_set.startswith("roadmap"):
        for chromhmm_key in _keys:
            # For 1_TssA, we need to add "1", "E1", and "TssA"
            number, mnemonic = chromhmm_key.split("_")
            CHROMHMM_STATES[annot_set][number] = CHROMHMM_STATES[annot_set][chromhmm_key]
            CHROMHMM_STATES[annot_set][f"E{number}"] = CHROMHMM_STATES[annot_set][chromhmm_key]
            CHROMHMM_STATES[annot_set][mnemonic] = CHROMHMM_STATES[annot_set][chromhmm_key]
            CHROMHMM_COLORS[annot_set][number] = CHROMHMM_COLORS[annot_set][chromhmm_key]
            CHROMHMM_COLORS[annot_set][f"E{number}"] = CHROMHMM_COLORS[annot_set][chromhmm_key]
            CHROMHMM_COLORS[annot_set][mnemonic] = CHROMHMM_COLORS[annot_set][chromhmm_key]


class ChromHMMSampleDataFrame(DataFrameBase):
    """
    A DataFrame of Public ChromHMM Samples
    """

    @classmethod
    def load_default_data(cls, *args, **kwargs):
        """Load the default chromhmm sample dataframe."""
        return cls(
            # This needs header=0 to use the column names from the dataframe,
            # otherwise it adds numbers and breaks things like self.clean_cell_name
            pandas.read_table(
                os.path.join(BASE_DIR, "./public_chromhmm.tsv"),
                sep="\t",
                header=0,
            ),
            *args,
            **kwargs,
        )

    def query_cell_types(self, cell_types, exact=False, exclude=False, case_sensitive=False):
        # We need to cast as list, otherwise loops over characters in a string
        if isinstance(cell_types, str):
            cell_types = [cell_types]

        if not case_sensitive:
            cell_types = [ct.lower() for ct in cell_types]

        def is_cell_type_match(clean_cell_name):

            if exclude:
                return not any([ct in clean_cell_name.lower() for ct in cell_types])
            elif exact:
                return any([ct == clean_cell_name.lower() for ct in cell_types])
            else:
                return any([ct in clean_cell_name.lower() for ct in cell_types])

        return self.loc[self.clean_cell_name.apply(lambda x: is_cell_type_match(x))]
