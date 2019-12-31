import logging
from typing import Tuple, Dict, Any, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .distance import EuclideanDistance, EuclideanMean, HyperbolicDistance, HyperbolicMean
except:  # noqa: E7222
    from distance import EuclideanDistance, EuclideanMean, HyperbolicDistance, HyperbolicMean


logger = logging.getLogger(__name__)


class RNNEncoder(nn.Module):
    """Implements a multi-layer RNN.

    This module can be used to create multi-layer RNN models, and
    provides a way to reduce to output of the RNN to a single hidden
    state by pooling the encoder states either by taking the maximum,
    average, or by taking the last hidden state before padding.

    Padding is delt with by using torch's PackedSequence.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule

    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 rnn_type: str = 'lstm',
                 dropout: float = 0,
                 bidirectional: bool = False,
                 layer_norm: bool = False,
                 highway_bias: float = 0,
                 rescale: bool = True,
                 enforce_sorted: bool = False,
                 **kwargs) -> None:
        """Initializes the RNNEncoder object.

        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        enforce_sorted: bool
            Whether rnn should enforce that sequences are ordered by
            length. Requires True for ONNX support. Defaults to False.
        kwargs
            Additional parameters to be passed to SRU when building
            the rnn.

        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`

        """
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        if rnn_type in ['lstm', 'gru']:
            if kwargs:
                logger.warn(f"The following '{kwargs}' will be ignored " +
                            "as they are only considered when using 'sru' as " +
                            "'rnn_type'")

            rnn_fn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_fn(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == 'sru':
            try:
                from sru import SRU
            except:  # noqa: E7222
                raise ImportError("SRU not installed. You can install it with: `pip install sru`")

            try:
                self.rnn = SRU(input_size,
                               hidden_size,
                               num_layers=n_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               layer_norm=layer_norm,
                               rescale=rescale,
                               highway_bias=highway_bias,
                               **kwargs)
            except TypeError:
                raise ValueError(f"Unkown kwargs passed to SRU: {kwargs}")
        else:
            raise ValueError(f"Unkown rnn type: {rnn_type}, use of of: gru, sru, lstm")

    def forward(self,  # type: ignore
                data: Tensor,
                state: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S]

        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x H]

        """
        data = data.transpose(0, 1)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1)

        if padding_mask is None:
            # Default RNN behavior
            output, state = self.rnn(data, state)
        elif self.rnn_type == 'sru':
            # SRU takes a mask instead of PackedSequence objects
            # Write (1 - mask_t) in weird way for type checking to work
            output, state = self.rnn(data, state, mask_pad=(-padding_mask + 1).byte())
        else:
            # Deal with variable length sequences
            lengths = padding_mask.long().sum(dim=0)
            # Pass through the RNN
            packed = nn.utils.rnn.pack_padded_sequence(data, lengths,
                                                       enforce_sorted=self.enforce_sorted)
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)

        output = output.transpose(0, 1).contiguous()

        # Compute lengths and pool the last hidden state before padding
        if padding_mask is None:
            lengths = torch.tensor([data.size(1)] * data.size(0)).long()
        else:
            lengths = padding_mask.long().sum(dim=1)

        return output[torch.arange(output.size(0)).long(), lengths - 1, :]


class PrototypicalTextClassifier(nn.Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    """

    def __init__(self,
                 vocab_size: int,
                 distance: str = 'euclidean',
                 embedding_dim: Optional[int] = None,
                 embedding_dropout: float = 0,
                 pretrained_embeddings: Optional[Tensor] = None,
                 freeze_pretrained_embeddings: bool = True,
                 padding_idx: int = 0,
                 hidden_dim: int = 128,
                 **kwargs) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        vocab_size: int
            The number of tokens in the vocabulary
        distance: str, optional
            One of: ['euclidean', 'hyperbolic']
        embedding_dim: int, optional
            The token embedding dimension. Should be provided if no
            pretrained embeddings are used.
        pretrained_embeddings: Tensor, optional
            A pretrained embedding matrix
        freeze_pretrained_embeddings: bool, optional
            Only used if a pretrained embedding matrix is provided.
            Freezes the embedding layer during training.
        padding_idx: int, optional
            The padding index. Default ``0``.
        hidden_dim: int, optional
            The hidden dimension of the encoder. Default ``128``.

       Extra keyword arguments are passed to the RNNEncoder.

        """
        super().__init__()

        if distance == 'euclidean':
            dist = EuclideanDistance()
            mean = EuclideanMean()
        elif distance == 'hyperbolic':
            dist = HyperbolicDistance()
            mean = HyperbolicMean()
        else:
            raise ValueError(f"Distance should be one of: ['euclidean', 'hyperbolic'], but got {distance}")

        self.distance_module = dist
        self.mean_module = mean
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        if embedding_dim is None and pretrained_embeddings is None:
            raise ValueError("At least one of: `embedding_dim` and `pretrained_embeddings` must be provided")

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                          freeze=freeze_pretrained_embeddings,
                                                          padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        kwargs['hidden_size'] = hidden_dim
        self.encoder = RNNEncoder(self.embedding.embedding_dim, **kwargs)

    def compute_prototypes(self, support: Tensor, label: Tensor) -> Tensor:
        """Set the current prototypes used for classification.

        Parameters
        ----------
        data : torch.Tensor
            Input encodings
        label : torch.Tensor
            Corresponding labels

        """
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])

        means = []
        n_means = len(means_dict)

        for i in range(n_means):
            # Ensure that all contiguous indices are in the means dict
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = self.mean_module(supports).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)

        prototypes = torch.stack(means, dim=0)
        return prototypes

    def forward(self,  # type: ignore
                query: Tensor,
                support: Optional[Tensor] = None,
                support_label: Optional[Tensor] = None,
                prototypes: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        query: Tensor
            The query examples, as tensor of shape (seq_len x batch_size)
        support: Tensor
            The support examples, as tensor of shape (seq_len x batch_size)
        support_label: Tensor
            The support labels, as tensor of shape (batch_size)

        Returns
        -------
        Tensor
            If query labels are

        """
        padding_mask = (query != self.padding_idx).byte()
        query_embeddings = self.embedding_dropout(self.embedding(query))
        query_encoding = self.encoder(query_embeddings, padding_mask=padding_mask)

        if prototypes is not None:
            prototypes = prototypes
        elif support is not None and support_label is not None:
            padding_mask = (support != self.padding_idx).byte()
            support_embeddings = self.embedding_dropout(self.embedding(support))
            support_encoding = self.encoder(support_embeddings, padding_mask=padding_mask)

            # Compute prototypes
            prototypes = self.compute_prototypes(support_encoding, support_label)
        else:
            raise ValueError("No prototypes set or provided")

        dist = self.distance_module(query_encoding, prototypes)
        return - dist
