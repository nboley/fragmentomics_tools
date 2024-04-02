import lightning as L
from torch import optim, nn

from layers import ResNetDilatedBlock
from loss import MultinomialNLLLoss, NegativeBinomialFixedTotalCountNLLLoss, NegativeBinomialNLLLossOld, NegativeBinomialNLLLoss

from fragmentomics_tools.plot.tracks import Tracks, StrandSplitCoverageTrack, VectorTrack, GeneTrack, BedTrack
from fragmentomics_tools.fragment_array import RegionFragmentArray, merge_fragment_arrays


# define the LightningModule
class BackgroundModelModule(L.LightningModule):
    def get_model(self):
        initial_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.model_params['input_channels'],
                out_channels=self.model_params['n_kernels'],  # one per strand so will be doubled
                kernel_size=self.model_params['kernel_size'],
                padding=0,
            ),
            nn.LeakyReLU(),
            SpatialDropout(p=self.model_params['dropout']),
        )
        
        res_net_layers = []
        res_net_layers.extend(
            [
                ResNetDilatedBlock(
                    dilation_rate=2**i,
                    input_channels=self.model_params['n_kernels'],
                    activation=nn.LeakyReLU,
                    profile_kernel_size=self.model_params['kernel_size'],
                    activation_post_sum=True,
                    skip_batchnorm=True,
                    preact_residual_normalization=False,
                    padding=0, # (self.model_params['kernel_size']-1),
                )
                for i in range(1, self.model_params['num_residual_layers']+1)
            ]
        )
        
        final_layer = nn.Conv1d(
            in_channels=self.model_params['n_kernels'],
            out_channels=self.output_tracks_multiplier*len(self.output_columns),
            kernel_size=(self.model_params['kernel_size'],),
            padding=0,
        )
        
        layers = [initial_layer] + res_net_layers + [final_layer]

        return nn.Sequential(*layers)

    def calc_input_region_size(self, output_region_size):
        return output_region_size \
        + 2*(self.model_params['kernel_size']-1)  \
        + sum([
            (self.model_params['kernel_size']-1)*(2**i) 
            for i in range(1, self.model_params['num_residual_layers']+1)
        ])

    def save(self, path):
        state_dict = self.model.state_dict()
        state_dict["model_params"] = self.model_params
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path):
        state_dict = torch.load(path)
        rv = cls(**state_dict.pop('model_params'))
        rv.model.load_state_dict(state_dict)
        return rv

    def __init__(
        self, 
        dropout = 0.15,
        input_channels = 4,
        output_columns = ['fwd_start_counts', 'bkwd_start_counts', 'fwd_stop_counts', 'bkwd_stop_counts'],
        num_residual_layers = 1,
        n_kernels = 512,
        kernel_size = 32,
        loss='multinomial',
    ):
        super().__init__()

        if loss == 'multinomial':
            self.loss_fn = MultinomialNLLLoss()
            self.output_tracks_multiplier = 1
        elif loss == 'negative_binomial':
            self.loss_fn = NegativeBinomialNLLLossOld()
            self.output_tracks_multiplier = 2
        elif loss == 'negative_binomial_fixed_total_count':
            self.loss_fn = NegativeBinomialFixedTotalCountNLLLoss(1)
            self.output_tracks_multiplier = 1
        else:
            raise ValueError(f"loss must be either 'multinomial' or 'negative_binomial' (saw '{loss}')")

        self.output_columns = output_columns

        self.model_params = {}
        self.model_params['loss'] = loss
        self.model_params['dropout'] = dropout
        self.model_params['output_columns'] = output_columns

        self.model_params['input_channels'] = input_channels
        self.model_params['num_residual_layers'] = num_residual_layers
        self.model_params['n_kernels'] = n_kernels
        self.model_params['kernel_size'] = kernel_size 

        # TODO -- fix these
        self.verbose = True
        self.num_workers = 64

        self.model = self.get_model()
        self.model.cuda()

    def plot_on_epoch_end(self, batch_idx):
        from fragmentomics_tools.fragment_array import RegionFragmentArray, merge_fragment_arrays
        GENE_NAME = 'FAM83E'
        sub_df = hge_raw.query("gene_name == @GENE_NAME")
        #merged_fa = merge_fragment_arrays([fa for fa in sub_df.fragment_array.tolist()])
        #merged_fa.region, merged_fa.n_fragments
        region = list(sub_df.iter_regions())[0]
        rfa = merge_fragment_arrays(SampleAndRegionDataFrame.init_from_rdf_and_sdf(sub_df, sdf).attach_fragment_arrays().fragment_array)
        self.predict_from_region_fragment_array(rfa).make_tracks().plot(title=GENE_NAME, out_fname=f"/scratch/karius/bias_correction_model/epoch_plots/{GENE_NAME}.batch_{batch_idx}.png")

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
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer