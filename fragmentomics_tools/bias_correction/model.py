from collections import defaultdict

import lightning as L
import torch
from torch import nn
import tqdm
import pandas as pd
import numpy as np
import matplotlib
from smart_open import open as smart_open
import io


from scipy.stats import nbinom, multinomial
from scipy.special import softmax

from .layers import ResNetDilatedBlock, SpatialDropout
from .loss import (
    MultinomialNLLLoss,
    NegativeBinomialFixedTotalCountNLLLoss,
    NegativeBinomialNLLLossOld,
    NegativeBinomialNLLLoss,
)
from .data import FragmentEndpointsDataset
from .predict import calc_stats, make_qc_plots, make_tracks

from fragmentomics_tools.plot.tracks import (
    Tracks,
    StrandSplitCoverageTrack,
    VectorTrack,
    GeneTrack,
    BedTrack,
)
from fragmentomics_tools.fragment_array import (
    RegionFragmentArray,
    merge_fragment_arrays,
)
from fragmentomics_tools.dataframe import (
    RegionDataFrame,
    SampleAndRegionDataFrame,
    DataFrameBase,
    windowed_range,
)
from fragmentomics_tools.region import one_hot_encode_sequences

# ignore a known warning
import warnings

warnings.filterwarnings("ignore", ".*Received a 3D input to dropout2d*")

FASTA_PATH = "/home/nboley/src/Ravel/data/repo_data_manifest/reference/GRCh38/GRCh38.p12.genome.fa.gz"


class BackgroundModelPredictionsStatsDataFrame(DataFrameBase):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, pd.core.internals.BlockManager):
            return

    def make_qc_plots(self):
        return make_qc_plots(self)


class BackgroundModelPredictionsDataFrame(RegionDataFrame):
    def build_stats_df(self):
        return BackgroundModelPredictionsStatsDataFrame(
            pd.DataFrame(self).progress_apply(calc_stats, axis=1).reset_index()
        )

    def make_tracks_for_gene(self, gene_name, *args, **kwargs):
        sub_df = self.query("gene_name == @gene_name")
        assert sub_df.shape[0] == 1
        return make_tracks(sub_df.iloc[0], *args, **kwargs)


# define the LightningModule
class BackgroundModelModule(L.LightningModule):
    def _build_dist(self, pred):
        rv = []
        for i in range(len(self.output_columns)):
            if self.model_params["loss"] == "negative_binomial":
                n, p = self.loss_fn.to_natural_param(pred[i], pred[i + 2])
                dist = nbinom(n=n, p=p)
            elif self.model_params["loss"] == "negative_binomial_old":
                ps = 1 - logits_to_probs(torch.Tensor(pred[i]), is_binary=True)
                total_counts = self.loss_fn._transform_counts(torch.Tensor(pred[i + 2]))
                dist = nbinom(n=total_counts, p=ps)
            elif self.model_params["loss"] == "negative_binomial_fixed_total_count":
                ps = 1 - logits_to_probs(torch.Tensor(pred[i]), is_binary=True)
                total_counts = self.loss_fn.total_count
                dist = nbinom(n=total_counts, p=ps)
            elif self.model_params["loss"] == "multinomiall":
                dist = multinomial(n=1, p=softmax(pred[i]))
            else:
                raise ValueError(f"Unrecognized loss '{self.model_params['loss']}'")
            rv.append(dist)
        return rv

    def get_model(self):
        initial_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.model_params["input_channels"],
                out_channels=self.model_params[
                    "n_kernels"
                ],  # one per strand so will be doubled
                kernel_size=self.model_params["kernel_size"],
                padding=0,
            ),
            torch.nn.LeakyReLU(),
            SpatialDropout(p=self.model_params["dropout"]),
        )

        res_net_layers = []
        res_net_layers.extend(
            [
                ResNetDilatedBlock(
                    dilation_rate=2**i,
                    input_channels=self.model_params["n_kernels"],
                    activation=torch.nn.LeakyReLU,
                    profile_kernel_size=self.model_params["kernel_size"],
                    activation_post_sum=True,
                    skip_batchnorm=True,
                    preact_residual_normalization=False,
                    padding=0,  # (self.model_params['kernel_size']-1),
                )
                for i in range(1, self.model_params["num_residual_layers"] + 1)
            ]
        )

        final_layer = torch.nn.Conv1d(
            in_channels=self.model_params["n_kernels"],
            out_channels=self.output_tracks_multiplier * len(self.output_columns),
            kernel_size=(self.model_params["kernel_size"],),
            padding=0,
        )

        layers = [initial_layer] + res_net_layers + [final_layer]

        return torch.nn.Sequential(*layers)

    def calc_input_region_size(self, output_region_size):
        return (
            output_region_size
            + 2 * (self.model_params["kernel_size"] - 1)
            + sum(
                [
                    (self.model_params["kernel_size"] - 1) * (2**i)
                    for i in range(1, self.model_params["num_residual_layers"] + 1)
                ]
            )
        )

    def save(self, path):
        state_dict = self.model.state_dict()
        state_dict["model_params"] = self.model_params
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path):
        with smart_open(path, "rb") as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer)

        rv = cls(**state_dict.pop("model_params"))
        rv.model.load_state_dict(state_dict)
        return rv

    def __init__(
        self,
        dropout=0.15,
        input_channels=4,
        output_columns=[
            "fwd_start_counts",
            "bkwd_start_counts",
            "fwd_stop_counts",
            "bkwd_stop_counts",
        ],
        num_residual_layers=1,
        n_kernels=512,
        kernel_size=32,
        loss="multinomial",
    ):
        super().__init__()

        if loss == "multinomial":
            self.loss_fn = MultinomialNLLLoss()
            self.output_tracks_multiplier = 1
        elif loss == "negative_binomial":
            self.loss_fn = NegativeBinomialNLLLossOld()
            self.output_tracks_multiplier = 2
        elif loss == "negative_binomial_fixed_total_count":
            self.loss_fn = NegativeBinomialFixedTotalCountNLLLoss(1)
            self.output_tracks_multiplier = 1
        else:
            raise ValueError(
                f"loss must be either 'multinomial' or 'negative_binomial' (saw '{loss}')"
            )

        self.output_columns = output_columns

        self.model_params = {}
        self.model_params["loss"] = loss
        self.model_params["dropout"] = dropout
        self.model_params["output_columns"] = output_columns

        self.model_params["input_channels"] = input_channels
        self.model_params["num_residual_layers"] = num_residual_layers
        self.model_params["n_kernels"] = n_kernels
        self.model_params["kernel_size"] = kernel_size

        # TODO -- fix these
        self.verbose = True
        self.num_workers = 64

        self.model = self.get_model()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y[:, :, :])
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # self.plot_on_epoch_end(batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y[:, :, :])
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def predict_from_seq(self, seq):
        self.eval()
        try:
            seq = seq.encode()
        except AttributeError:
            assert isinstance(seq, bytes)

        x = one_hot_encode_sequences([seq])[0].T
        return self.model(torch.Tensor(x).to(self.device)).cpu().detach().numpy()

    def predict_from_fasta(self, fa, max_region_size=100000):
        # calculate how much context each region needs to make a prediction
        region_expansion = (
            self.calc_input_region_size(max_region_size) - max_region_size
        )

        # build a list of all regions split by max_region_size
        regions = []
        for contig, length in zip(fa.references, fa.lengths):
            for start, stop in windowed_range(
                0, length - region_expansion, max_region_size
            ):
                regions.append((contig, start, stop + region_expansion))

        # predict on each window
        windowed_pred = defaultdict(list)
        for contig, start, stop in tqdm.tqdm(regions):
            windowed_pred[contig].append(
                self.predict_from_seq(fa.fetch(contig, start, stop))
            )

        # merge the windows together, and build the dists
        dists = []
        for contig, length in zip(fa.references, fa.lengths):
            dist = self._build_dist(np.hstack(windowed_pred[contig]))
            output_length = dist[0].kwds["p"].shape[0]
            start = (length - output_length) // 2
            end = output_length + start
            dists.append([contig, start, end] + dist)

        return pd.DataFrame(
            dists,
            columns=["contig", "start", "stop"]
            + [c + "_dist" for c in self.output_columns],
        )

    def predict_from_rdf(self, rdf):
        rdf = rdf.resize_regions(self.calc_input_region_size(rdf.region_lengths))
        seq = rdf.get_one_hot_encoded_sequence(FASTA_PATH, verbose=True)

        self.eval()
        pred = seq.progress_apply(
            lambda x: self.model(torch.Tensor(x).to(self.device)).cpu().detach().numpy()
        )
        dists = [self._build_dist(x) for x in tqdm.tqdm(pred)]

        return pd.DataFrame(
            dists, columns=[x + "_dist" for x in self.output_columns], index=rdf.index
        )

    def predict_from_rdf_and_sdf(self, rdf, sdf):
        print("Predicting dists (1/5)")
        pred_dists = self.predict_from_rdf(rdf.set_index("gene_id"))

        print("Building fragment arrays (2/5)")
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(
            rdf, sdf
        ).attach_fragment_arrays(num_cores=32, min_mapq=10)
        print("Merging fragment arrays (3/5)")
        merged_fas = (
            srdf.groupby("gene_id")
            .progress_apply(
                lambda x: merge_fragment_arrays(
                    [fa.drop_duplicate_fragments() for fa in x.fragment_array]
                )
            )
            .rename("fragment_array")
        )
        print("Building observed arrays (4/5)")

        fas_and_blacklist_regions = (
            pd.DataFrame(rdf)
            .set_index("gene_id")[["blacklist_regions"]]
            .join(merged_fas)
        )
        print("Building observed arrays (4/5)")

        obs_counts = fas_and_blacklist_regions.progress_apply(
            lambda x: pd.Series(
                list(
                    FragmentEndpointsDataset.get_targets_from_region_fragment_array(
                        x.fragment_array, self.output_columns, x.blacklist_regions
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            ),
            axis=1,
        )
        obs_counts = pd.DataFrame(obs_counts).rename(
            columns=dict(enumerate(self.output_columns))
        )

        print("Merging Results (5/5)")
        return BackgroundModelPredictionsDataFrame(
            rdf.set_index("gene_id")
            .join(obs_counts.join(pred_dists, how="inner"))
            .reset_index(),
            ref=rdf.ref,
        )
