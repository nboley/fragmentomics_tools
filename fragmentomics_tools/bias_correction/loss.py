import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.utils import logits_to_probs, probs_to_logits


def _loss_reduce(nll, reduction):
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise NotImplementedError(f"Reduction {reduction} is not implemented.")


class MultinomialNLLLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        is_probs=False,
        reduction: str = "mean",
        scale_loss_to_d: bool = False,
    ) -> None:
        """Multinomial Negative Log Likelihood Loss

        :param size_average: (bool, optional) Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        :param reduce: (bool, optional) Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        :param reduction:  (string, optional) Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        :param scale_loss_to_d: if true, loss is scaled by 1/sqrt(dimension)
        :param is_probs: (bool). Set to true if you have the model output probabilities rather than
            logits. This should usually be false.
        """
        super().__init__(size_average, reduce, reduction)
        self.scale_loss_to_d = scale_loss_to_d
        self.is_probs = is_probs

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = input.shape
        # *dims[:2] is batch, channel
        inputs_flat = input.contiguous().view(*dims[:2], -1)
        target_flat = target.contiguous().view(*dims[:2], -1)
        if self.is_probs:
            # Note total_count is not used when calculating log_prob, and for log_prob,
            # validate_args is just a check that throws errors if the sum doesn't match total_count
            dist = torch.distributions.Multinomial(
                probs=inputs_flat, validate_args=False
            )
        else:
            dist = torch.distributions.Multinomial(
                logits=inputs_flat, validate_args=False
            )

        if self.scale_loss_to_d:
            d = inputs_flat.shape[-1]
            scale = 1.0 / sqrt(d)
        else:
            scale = 1.0
        with torch.no_grad():
            min_counts, _ = target_flat.min(dim=-1)
            min_count = min_counts.min()
            if min_count < 0:
                raise ValueError(f"Saw a count < 0: {min_count}, all mins:{min_counts}")

        nll = -dist.log_prob(target_flat) * scale
        return _loss_reduce(nll, self.reduction)


def convert_params(mu, alpha):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    var = mu + alpha * mu**2
    p = mu / var
    n = mu**2 / (var - mu)
    return n, p


class NegativeBinomialNLLLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduction: str = "mean") -> None:
        """Multinomial Negative Log Likelihood Loss


        :param reduction:  (string, optional) Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        """
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    @staticmethod
    def to_natural_param(param_1_output, param_2_output):
        means = torch.sigmoid(torch.Tensor(param_1_output) - 2)
        dispersion = torch.exp(-(torch.Tensor(param_2_output) - 6))
        return convert_params(means, dispersion)  # returns n, p

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        means_input = torch.sigmoid(input[:, :target.shape[1], :] - 2)
        # expected input is r, or total count, which is >0
        # we let the model fit log(alpha = 1/r) = -log(r)
        # => exp(x) = 1/r => r = exp(-x)
        dispersion_input = torch.exp(-(input[:, target.shape[1]:, :] - 6))

        dims = means_input.shape
        assert dims == dispersion_input.shape
        means_input_flat = means_input.contiguous().view(*dims[:2], -1)
        dispersion_input_flat = dispersion_input.contiguous().view(*dims[:2], -1)
        target_flat = target.contiguous().view(*dims[:2], -1)

        n, p = convert_params(means_input_flat, dispersion_input_flat)

        dist = torch.distributions.NegativeBinomial(
            total_count=n, probs=p, validate_args=False
        )

        nll = -dist.log_prob(target_flat)
        return _loss_reduce(nll, self.reduction)


class NegativeBinomialNLLLossOld(torch.nn.modules.loss._Loss):
    def __init__(self, reduction: str = "mean", min_alpha=0.0, max_alpha=0.05) -> None:
        """Multinomial Negative Log Likelihood Loss


        :param reduction:  (string, optional) Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        """
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def to_natural_param(self, param_1_output, param_2_output):
        n = self.min_alpha + self.max_alpha * torch.sigmoid(
            torch.Tensor(param_2_output)
        )
        p = 1 - logits_to_probs(torch.Tensor(param_1_output), is_binary=True)
        return n, p

    @staticmethod
    def transform_counts(min_alpha, max_alpha, total_count_flat):
        return min_alpha + max_alpha * torch.sigmoid(total_count_flat)

    def _transform_counts(self, total_count_flat):
        return self.transform_counts(self.min_alpha, self.max_alpha, total_count_flat)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_input = input[:, :target.shape[1], :]
        total_count_input = input[:, target.shape[1]:, :]
        dims = logits_input.shape
        assert dims == total_count_input.shape

        logits_flat = logits_input.contiguous().view(*dims[:2], -1)
        total_count_flat = self._transform_counts(
            total_count_input.contiguous().view(*dims[:2], -1)
        )
        target_flat = target.contiguous().view(*dims[:2], -1)

        dist = torch.distributions.NegativeBinomial(
            total_count=total_count_flat, logits=logits_flat, validate_args=False
        )

        nll = -dist.log_prob(target_flat)
        return _loss_reduce(nll, self.reduction)


class NegativeBinomialFixedTotalCountNLLLoss(torch.nn.modules.loss._Loss):
    def __init__(self, total_count, reduction: str = "mean") -> None:
        """Multinomial Negative Log Likelihood Loss


        :param reduction:  (string, optional) Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        """
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        self.total_count = total_count

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_input = input[:, :2, :]
        dims = logits_input.shape

        logits_flat = logits_input.contiguous().view(*dims[:2], -1)
        target_flat = target.contiguous().view(*dims[:2], -1)

        dist = torch.distributions.NegativeBinomial(
            total_count=self.total_count, logits=logits_flat, validate_args=False
        )

        nll = -dist.log_prob(target_flat)
        return _loss_reduce(nll, self.reduction)
