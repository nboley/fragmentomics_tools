import numpy
import scipy

from torch import nn, Tensor

from fragmentomics_tools.region import Region


def calculate_start_and_stop_from_jitter(
    input_length, output_length, jitter_value, strand: Optional[str] = None
):
    """
    Calculates the new start and stop of a region given some +/- jitter value and output length
    Importantly, this is only defined on input / output lengths of the same parity:
        allowed: (even, even), (odd, odd)
        disallowed: (even, odd), (odd, even)
    :param input_length: length of input region
    :param output_length: desired output length of region
    :param jitter_value: a +/- value from the center of the input region by which to shift window
    :param strand: Optional strand, if not supplied this will default to assuming the user wants + strand.
    :return: coordinates for, e.g., slicing
    """
    if input_length < output_length + numpy.abs(jitter_value):
        raise ValueError(
            f"input array is not wide enough to allow for an output of"
            f" length {output_length} with jitter_value of {jitter_value}."
        )

    resize_start = Region.get_resize_start(
        start=0, current_size=input_length, new_size=output_length, strand=strand
    )
    jitter_start = resize_start + jitter_value
    jitter_stop = jitter_start + output_length
    return jitter_start, jitter_stop

def jitter_matrix(input_arr, jitter_value, output_length, strand: Optional[str] = None):
    input_length = input_arr.shape[-1]

    if round(jitter_value) != jitter_value:
        raise ValueError(f"jitter_range must be a whole number, received {jitter_value}.")

    if input_length < output_length + numpy.abs(jitter_value):
        raise ValueError(
            f"input array is not wide enough to allow for an output of"
            f" length {output_length} with jitter_value of {jitter_value}."
        )

    new_start, new_stop = calculate_start_and_stop_from_jitter(
        input_length=input_length, output_length=output_length, jitter_value=jitter_value, strand=strand
    )
    assert new_stop - new_start == output_length, (output_length, new_stop, new_start)
    assert new_start >= 0, new_start
    assert new_stop <= input_length, (new_stop, input_length)

    if isinstance(input_arr, numpy.ndarray) or isinstance(input_arr, torch.Tensor):
        jittered_output = input_arr[..., new_start:new_stop]
    elif scipy.sparse.issparse(input_arr):
        indices = numpy.where((input_arr.col < new_stop) & (input_arr.col >= new_start))[0]

        jittered_output = coo_matrix(
            (
                numpy.ones(len(indices)),
                (input_arr.row[indices], input_arr.col[indices] - new_start),
            ),
            shape=[input_arr.shape[0], output_length],
            dtype=input_arr.dtype,
        )
    else:
        raise TypeError(f"input_arr type {type(input_arr)} is invalid.")

    return jittered_output


class SpatialDropout(nn.Module):
    def __init__(self, p=0.2):
        """TLDR; Full Channel Dropout in ND. This replicates keras SpatialDropout1D but with the pytorch expected
            channel ordering. See https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/4
            Basically this will drop out full channels, or leave them in tact.
        :param p: dropout fraction, also note that the non-dropped values are filled with 1/p as in pytorch.
        """
        super().__init__()
        self.p = p
        self.do2 = nn.Dropout2d(p)

    def forward(self, x):
        if self.training:
            ori_shape = x.shape
            x = x.view(*ori_shape[:2], -1)  # flatten everything beyond the channel dimension
            x = x.permute(0, 2, 1)
            x = self.do2(x)
            x = x.permute(0, 2, 1)
            x = x.view(
                ori_shape
            )  # restore the feature shape, pixels will have been dropped out across channels.
        return x


class ResNetDilatedBlock(nn.Module):
    def __init__(
        self,
        input_channels: int = 64,
        profile_kernel_size: int = 20,
        dilation_rate: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        activation_post_sum: bool = False,
        skip_batchnorm: bool = False,
        preact_residual_normalization: bool = False,
        padding: Union[str, int] = "same",
    ):
        """A Dilated Convolutional Block with user defined activation and padding so input size == output size.
            x is added back to the output after passing through the dilated convolution layer.

            This Module does not change the inputs dimensionality.

        :param input_channels: The number of filters (in the conv in filters == out filters).
        :param profile_kernel_size: Profile kernel size.
            This should be odd unless the dilation rate is even, then it can be either.
        :param dilation_rate: rate of dilation
        :param activation: module of desired activation function. Defaults to nn.ReLU.
        :param preact_residual_normalization: Apply a pre-activation layer (recommended) to limit
            accumulation of variance in deeper networks. See:
            https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#moment-control
        :param activation_post_sum: Apply activation after adding x back in (the residual connection). Default=True
            which is what pytorch does in it's resnet implementation.
        """
        super().__init__()
        assert (
            profile_kernel_size % 2 == 1 or dilation_rate % 2 == 0
        ), f"Please provide an odd kernel size, you gave {profile_kernel_size}"
        self.activation_post_sum = activation_post_sum
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            stride=1,
            kernel_size=profile_kernel_size,
            padding=(dilation_rate * (profile_kernel_size - 1)) // 2 if padding == "same" else padding,
            dilation=dilation_rate,
        )
        self.preact_residual_normalization = preact_residual_normalization
        if self.preact_residual_normalization:
            self.bn_preact = nn.BatchNorm1d(input_channels)
        else:
            self.bn_preact = None
        self.bn = nn.BatchNorm1d(input_channels)
        self.skip_batchnorm = skip_batchnorm
        self.activation = activation()

    def forward(self, x: Tensor):
        """
        :param x: Tensor
        :return: Tensor of the same shape as input.
        """
        if self.preact_residual_normalization:
            # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#moment-control
            # where preact is later defined as nn.Sequential([norm_layer(inplanes), nn.ReLU()])
            # in PreResidualBottleneck. The alpha/beta terms are stored inside of the norm_layer.
            assert self.bn_preact is not None
            x = self.activation(self.bn_preact(x))
        out = self.conv1(x)
        if not self.skip_batchnorm:
            out = self.bn(out)
        if out.shape[-1] < x.shape[-1]:
            x = jitter_matrix(x, jitter_value=0, output_length=out.shape[-1])
        if self.activation_post_sum:
            # default pytorch method
            resid = out + x
            return self.activation(resid)
        else:
            out = self.activation(out)
            return out + x

