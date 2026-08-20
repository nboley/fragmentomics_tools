import pysam

from typing import Optional, Dict, Union, Any, List, Set

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from fragmentomics_tools.public_data_resources.jaspar import (
    BindingModel,
    MissingPFMError,
    get_all_human_pfms,
)
from sequence import one_hot_encode_sequences
# logger = logging.getLogger(__name__)


class Pfm:
    @staticmethod
    def entropy(x):
        ent = np.zeros_like(x)
        valid_pos = x != 0
        ent[valid_pos] = -x[valid_pos] * np.log2(x[valid_pos])
        return ent

    def __init__(self, freqs: np.ndarray, pseudocount: int = 0):
        assert freqs.ndim == 3
        self.num_regions, self.length, self.num_chars = freqs.shape
        self.freqs = freqs + pseudocount

        assert self.num_chars in {4, 5}, f"Invalid number of characters: {self.num_chars}"
        assert (
            (self.freqs == 0) | (self.freqs == 1)
        ).all(), "Some positions in one-hot matrix are not 0 or 1"

        self.colors = {
            "A": "xkcd:green",
            "C": "xkcd:blue",
            "G": "xkcd:orange",
            "T": "xkcd:red",
            "N": "xkcd:grey",
        }
        self.base_map = dict(list(zip("ACGTN", list(range(5)))) + list(zip(list(range(5)), "ACGTN")))

        for ii, ll in enumerate("ACGTN"):
            self.colors[ii] = self.colors[ll]

        self.pwm = self.freqs.mean(axis=0)

        # H_ib: Information at position i for base b
        self.H_ib = self.entropy(self.pwm)

        # H_i: Shannon entropy at position i
        self.H_i = self.H_ib.sum(axis=1)

        # e_n: Smoothing factor for small samples
        self.e_n = 1 / np.log(2) * (4 - 1) / (2 * self.num_regions)

        # R_i: Information content at position i
        self.R_i = 2 - (self.H_i + self.e_n)  # log2(4) - (H_i + e_n)

        # Total information content
        self.R = np.sum(self.R_i)

        # The height of each letter in the sequence
        self.seq_heights = self.freqs.sum(axis=0) * self.R_i[:, None]

    def __len__(self):
        return self.pwm.shape[0]

    def __str__(self):
        return str(self.freqs.sum(axis=0))

    def __repr__(self):
        return f"{repr(self.__class__)}: {repr(self.freqs.sum(axis=0))}"

    @property
    def sequence_logo(self):
        # This is normalized to bit space
        return self.seq_heights / self.num_regions

    @property
    def consensus_logo(self):
        return self.seq_heights.argmax(axis=1), self.seq_heights.max(axis=1)

    @property
    def score(self):
        return self.R

    def plot(self):
        from fragmentomics_tools.plot.tracks import MotifTrack
        from fragmentomics_tools.region import Region
        return MotifTrack(self.sequence_logo, region=Region("NA", 0, len(self)), rel_width=1.0).plot()


def get_pfms(tfs, all_pfms=None, default_jaspar: bool = False) -> List[BindingModel]:
    fallback_db = "HOCOMOCO" if default_jaspar else "JASPAR CORE"
    if all_pfms is None:
        # Note: the following command returns manually overrides on some TFs when we
        #  hand curate a different PFM for a particular TF name. This happens regardless
        #  of whether hocomoco is set to true or false.
        all_pfms = get_all_human_pfms(calculate_logo=True, hocomoco=not default_jaspar)
        primary_source = ("JASPAR CORE" if default_jaspar else "HOCOMOCO") + " (+ manual overrides)"
    else:
        primary_source = "the caller-supplied 'all_pfms'"
    all_pfms = list(all_pfms)
    pfms = {
        h["pfm_name"]: BindingModel(h["pfm"], id=h["pfm_id"], name=h["pfm_name"])
        for h in all_pfms
        if h["pfm_name"] in tfs
    }
    missing_tfs = set(tfs) - set(pfms.keys())
    if len(missing_tfs) > 0:
        # Try the other database as fallback. In containers the fallback DB
        # file may not exist — treat that as an empty fallback and let
        # MissingPFMError report the unresolved TFs.
        try:
            fallback_all_pfms = list(get_all_human_pfms(calculate_logo=True, hocomoco=default_jaspar))
        except FileNotFoundError:
            fallback_all_pfms = []
        fallback_pfms = {
            h["pfm_name"]: BindingModel(h["pfm"], id=h["pfm_id"], name=h["pfm_name"])
            for h in fallback_all_pfms
            if h["pfm_name"] in missing_tfs
        }
        for k, v in fallback_pfms.items():
            pfms[k] = v
        unresolved = set(tfs) - set(pfms.keys())
        if len(unresolved) > 0:
            raise MissingPFMError(
                f"No PFM could be found for {len(unresolved)} of the {len(set(tfs))} requested "
                f"TF(s): {sorted(unresolved)}. Searched {len(all_pfms)} PFMs from "
                f"{primary_source} and {len(fallback_all_pfms)} PFMs from the {fallback_db} "
                f"fallback."
            )
    return list(v for _, v in sorted(pfms.items(), key=lambda kv: kv[0]))  # sorted by tf name


class RegionDataset(torch.utils.data.Dataset):
    def __init__(self, region_dataframe, seed: Optional[int] = None):
        """
        This dataset is a superclass of probably all Datasets the functional genomics team will use
        """
        super().__init__()

        # there are nice rules for defining paths etc, so we can use a pattern with properties where
        #  the user doesn't know these are process specific
        self.region_dataframe = region_dataframe
        self.seed = np.random.randint(32767) if seed is None else seed
        self.rngs: Dict[int, np.random.RandomState] = {}

    @property
    def worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return -1  # arbitrary number for the main worker
        return worker_info.id

    @property
    def rng(self) -> np.random.RandomState:
        # Get a random number generator that is offset for the calling worker.
        wid = self.worker_id
        if wid not in self.rngs:
            self.rngs[wid] = np.random.RandomState(self.seed + wid + 1)
        return self.rngs[wid]

    def __getitem__(self, index):
        record = self.region_dataframe.iloc[index]
        return record

    def __len__(self):
        return len(self.region_dataframe)


class SeqDataSet(RegionDataset):
    def __init__(
        self,
        region_dataframe,
        ref_path: str,
        seed: Optional[int] = None,
        output_dict: bool = False,
        include_gc_if_dict: bool = False,
        include_strand_bit: bool = False,
        gc_window: int = 101,
    ):
        """This dataset example is a simple/useful one that returns a one-hot encoded sequence
            Feel free to use it as a template.

        :param region_dataframe: Regions dataframe from which to create seqs, make sure all regions are the same size
        :param seed: random seed to initialize the generators rng.
        :param output_dict: if set to true, a dictionary of named arrays is output. This is
            expected by some models. If false a single array is output. This must be true for outputting
            a gc track though, if desired. The name of the one-hot sequence tensor will be
            'input_sequence'.
        :param include_gc_if_dict: If output_dict is true, include a gc track in the output. The
            name for this track will be 'input_gc'
        :param include_strand_bit: If output_dict is true, include a 0, 1 bit to represent + or - strand
            input regions respectively. The name of this track will be 'minus_strand'
        :param gc_window: Window size for calculating local gc content, only used if the gc track
            is requested.
        Returns:
            __getitem__ will return either a single numpy array, or a dictionary of arrays, depending
            on settings.
        """
        super().__init__(region_dataframe, seed)
        self.ref_path = ref_path
        self.output_dict = output_dict
        self.include_gc_if_dict = include_gc_if_dict
        self.gc_window = gc_window
        self.include_strand_bit = include_strand_bit
        if self.include_gc_if_dict:
            self.gc_smoother = SequenceOneHotToWindowGcFraction(window_width=gc_window)
        else:
            self.gc_smoother = None

        # there are nice rules for defining paths etc, so we can use a pattern with properties where
        #  the user doesn't know these are process specific
        self.fasta_files: Dict[int, pysam.FastaFile] = {}

    @property
    def fasta_file(self) -> pysam.FastaFile:
        wid = self.worker_id
        if wid not in self.fasta_files:
            self.fasta_files[wid] = pysam.FastaFile(
                filename=self.ref_path, filepath_index_compressed=self.ref_path + ".gzi"
            )
        return self.fasta_files[wid]

    def __getitem__(self, index) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        # Select the requested record
        record = self.region_dataframe.iloc[index]
        contig = record.contig
        start = record.start
        stop = record.stop
        strand = "+" if "strand" not in dir(record) else record.strand
        # input sequence
        seq = one_hot_encode_sequences([self.fasta_file.fetch(contig, start, stop).encode()]).astype(
            "float32"
        )
        # first axis is empty
        seq = np.swapaxes(seq[0], 1, 0)
        if self.output_dict:
            data = {
                "input_sequence": seq,
            }
            if self.include_gc_if_dict:
                data["input_gc"] = self.gc_smoother(torch.tensor(seq[None, :]))[0].numpy()
            if self.include_strand_bit:
                data["minus_strand"] = np.array([1 if strand == "-" else 0], dtype=np.float)
            return data
        else:
            return seq


class TFConv1D(nn.Module):
    def __init__(
        self,
        channels=4,
        tf_width=11,
        stride=1,
        padding=0,
        normalize: bool = False,
        logo_weight: bool = False,
        best_from_logo: bool = True,
        only_logo: bool = True,
        output_both_strands: bool = False,
        all_hocomoco: Optional[List] = None,
        shuffle_motifs: bool = False,
        default_jaspar: bool = False,
        shuffle_seed: int = 314,
        tfs: Set[str] = None,
        pfms: List[BindingModel] = None,
        only_right_pad: bool = False,
    ):
        """Apply one of the profile smoothing options to a 1d array with an additional 1d channel first.

        :param channels: channels of input/output.
            The requested smoothing operation is applied independently to each channel.
        :param smoothing_width: width of smoothing window to be applied over data
        :param kind: "gaussian", "sum", or "mean".
        :param output_both_strands: If you want both strands rather than just the max scoring one.
        :param gaussian_sigma: ignored unless kind = "gaussian"
        :param only_right_pad: align the pwm on the left side of the filter so that we can subset later on
        """
        super().__init__()
        if pfms is not None:
            assert tfs is None
            self.pfms = pfms
        elif tfs is not None:
            self.pfms: List[BindingModel] = get_pfms(
                tfs, all_pfms=all_hocomoco, default_jaspar=default_jaspar
            )
        else:
            raise ValueError("Either 'pfms' or 'tfs' must be set.")
        if len(self.pfms) == 0:
            raise ValueError(
                f"TFConv1D requires at least one PFM, but resolved 0. "
                f"Requested tfs={tfs!r}, pfms={pfms!r}."
            )
        assert tf_width % 2 == 1, f"Only support odd tf lengths currently, got {tf_width}"
        self.tf_width = tf_width
        self.stride = stride
        self.channels = channels
        self.padding = padding
        self.shuffle_motifs = shuffle_motifs
        self.output_both_strands = output_both_strands
        self.default_jaspar = default_jaspar
        motif_tensors = []
        motif_tensors_rc = []
        with torch.no_grad():
            if self.shuffle_motifs:
                rng = np.random.RandomState(shuffle_seed)
                self.shuffle = rng.permutation(tf_width)
                self.shuffle_rc = np.copy(self.shuffle[::-1])
            else:
                self.shuffle = None
                self.shuffle_rc = None
            for pfm in self.pfms:
                logo = pfm.logo
                if best_from_logo:
                    start, end = self.get_best_score(logo, l=tf_width)
                else:
                    start, end = self.get_best_score(pfm, l=tf_width)
                logo_t = torch.tensor(
                    np.swapaxes(logo[start:end], 1, 0), dtype=torch.float32, requires_grad=False
                )
                logo_tr = torch.tensor(
                    np.swapaxes(logo[start:end][::-1, ::-1].copy(), 1, 0),
                    dtype=torch.float32,
                    requires_grad=False,
                )
                pfm_t = torch.tensor(
                    np.swapaxes(pfm[start:end], 1, 0), dtype=torch.float32, requires_grad=False
                )
                pfm_tr = torch.tensor(
                    np.swapaxes(pfm[start:end][::-1, ::-1].copy(), 1, 0),
                    dtype=torch.float32,
                    requires_grad=False,
                )
                if normalize and not only_logo:
                    pfm_t /= pfm_t.sum()
                    pfm_tr /= pfm_tr.sum()
                if normalize and only_logo:
                    logo_t /= logo_t.sum()
                    logo_tr /= logo_tr.sum()
                if only_logo:
                    motif_t = logo_t
                    motif_tr = logo_tr
                else:
                    if logo_weight:
                        motif_t = pfm_t * logo_t
                        motif_tr = pfm_tr * logo_tr
                    else:
                        motif_t = pfm_t
                        motif_tr = pfm_tr
                assert motif_t.shape[0] == 4
                if end - start < tf_width:
                    w = end - start
                    if only_right_pad:
                        n_pad_right = tf_width - w
                        n_pad_left = 0
                    else:
                        n_pad_right = (tf_width - w) // 2
                        n_pad_left = n_pad_right + 1 if w % 2 == 0 else n_pad_right
                    motif_t = F.pad(motif_t, (n_pad_left, n_pad_right))
                    # Derive RC from the padded forward kernel so padding
                    # position is automatically correct for both strands
                    motif_tr = motif_t.flip(dims=(-1, -2))
                assert motif_t.shape[0] == 4, motif_t.shape
                assert motif_t.shape[1] == tf_width, (motif_t.shape, w, tf_width)
                motif_tensors.append(motif_t)
                motif_tensors_rc.append(motif_tr)
            motifs = torch.stack(motif_tensors, dim=0)
            motifs_rc = torch.stack(motif_tensors_rc, dim=0)
            if self.shuffle is not None:
                assert self.shuffle_rc is not None
                motifs = motifs[..., self.shuffle].contiguous()
                motifs_rc = motifs_rc[..., self.shuffle_rc].contiguous()
            assert motifs.shape[1] == 4, (motifs.shape, motifs_rc.shape)
            assert motifs.shape[2] == tf_width, (motifs.shape, motifs_rc.shape, tf_width)
            assert motifs.shape[0] == len(self.pfms), (motifs.shape, motifs_rc.shape, len(self.pfms))
        self.register_buffer("kernel_f", motifs.clone().detach().requires_grad_(False))
        self.register_buffer("kernel_rc", motifs_rc.clone().detach().requires_grad_(False))

    @property
    def tf_names(self) -> List[str]:
        return [bm.name for bm in self.pfms]

    @property
    def tf_ids(self) -> List[str]:
        return [bm.pwm_id for bm in self.pfms]

    @property
    def pfm_lengths(self) -> List[int]:
        return [len(pfm) for pfm in self.pfms]

    @property
    def max_tf_scores(self) -> torch.Tensor:
        """Return maximum possible tf scores"""
        # self.kernel_f (n_tfs x 4 x tf_width)
        per_pos_max = torch.max(self.kernel_f, dim=1)[0]  # (n_tfs x tf_width)
        return per_pos_max.sum(-1)  # n_tfs

    @staticmethod
    def get_best_score(logo, l=11):
        if len(logo) <= l:
            return (0, len(logo))
        else:
            starts_scores = [(s, np.sum(logo[s : s + l])) for s in range(len(logo) - l)]
            best = sorted(starts_scores, key=lambda x: x[1])[-1]
            s = best[0]
            return (s, s + l)

    def forward(self, x):
        # p1d = (0, self.tf_width - 1)
        # x_pad = F.pad(x, p1d, mode="constant", value=0)
        f_conv = F.conv1d(x, self.kernel_f, padding=self.padding, stride=self.stride)
        rc_conv = F.conv1d(x, self.kernel_rc, padding=self.padding, stride=self.stride)
        if self.output_both_strands:
            # add a new dimension after channel, collapse along that
            return torch.cat((f_conv[:, :, None], rc_conv[:, :, None]), dim=2)
        else:
            max_strand = torch.maximum(f_conv, rc_conv)
            return max_strand

    def plot_logo(self, tf_name: str = "CTCF", length: int = 11, axes=None):
        pwm: BindingModel = {bm.name: bm for bm in self.pfms}[tf_name]
        logo = pwm.logo

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(12, 4))
        else:
            assert len(axes) == 2
        start, end = self.get_best_score(logo, l=length)
        axes[0].set_title(f"TF={tf_name}, ID={pwm.id}, Forward")
        plot_weights_given_ax(axes[0], logo, highlight={"red": [(start, end)]})
        axes[1].set_title(f"TF={tf_name}, ID={pwm.id}, Reverse")
        logor = logo[::-1, ::-1]
        startr, endr = self.get_best_score(logor, l=length)
        plot_weights_given_ax(axes[1], logor, highlight={"red": [(startr, endr)]})
