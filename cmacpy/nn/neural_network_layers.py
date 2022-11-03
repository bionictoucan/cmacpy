import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Dict, Union


class FCLayer(nn.Module):
    """
    A modifiable fully-connected layer for deep networks.

    Parameters
    ----------
    in_nodes : int
        The number of input nodes to the fully-connected layer.
    out_nodes : int
        The number of output nodes of the fully-connected layer.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is ``None`` -- no
        normalisation used. Options ``"batch"``, ``"instance"``, ``"group"`` and
        ``"layer"`` are supported to perform batch, instance, group or layer normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is ``"relu"`` -- uses the
        :abbr:`ReLU (rectified linear unit)` nonlinearity. Options supported are
            * ``"relu"`` -- a piecewise function where positive values are
              mapped to themselves and negative values are mapped to 0. This can
              be written as :math:`\\textrm{ReLU}(x) = \\textrm{max}(0, x)`,

            * ``"leaky_relu"`` -- a variant of the ReLU where negative values
              are mapped to :math:`\\alpha x` where :math:`\\alpha < 1`. The
              default value is 0.01 but this gradient can be set using the
              ``act_kwargs`` dictionary by setting ``act_kwargs =
              {"negative_slope" : value}``. Mathematically, this can be written

              .. math:: 
                  \\textrm{Leaky ReLU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha x),

            * ``"sigmoid"`` -- the sigmoid function
              
              .. math::
                  \\textrm{Sigmoid} (x) = \\frac{1}{1 + e^{-x}},

            * ``"tanh"`` -- the hyperbolic tangent function
              
              .. math::
                  \\textrm{tanh} (x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}},

            * ``"elu"`` -- the exponential linear unit is similar to the Leaky
              ReLU but the negative values are mapped to a scaled exponential
              curve
              
              .. math::
                  \\textrm{ELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x} - 1)),
                                      
              where :math:`\\alpha` defaults to 1.0 and can be assigned using
              the ``act_kwargs = {"alpha" : value}``,

            * ``"prelu"`` -- parameterised ReLU. This performs the same
              transformation as the Leaky ReLU but the :math:`\\alpha` parameter
              is learned. The starting value for :math:`\\alpha` is 0.25 but
              this can be changed by doing ``act_kwargs = {"init" : value}``,

            * ``"rrelu"`` -- randomised ReLU. Another variant of the Leaky ReLU
              where the value of :math:`\\alpha` is sampled from the uniform
              distribution :math:`\\mathcal{U} (\\textrm{lower},
              \\textrm{upper})` where the default value for lower is 1/8 and the
              default value for upper is 1/3. These can be set by ``act_kwargs =
              {"lower" : value1, "upper" : value2}``,

            * ``"selu"`` -- scaled ELU. Performs the ELU transformation
              multiplied by a fixed scaling factor
              
              .. math::
                  \\textrm{SELU} (x) = s * \\textrm{ELU} (x; \\alpha),
                  
              where :math:`s` = 1.0507009873554804934193349852946 and
              :math:`\\alpha` = 1.6732632423543772848170429916717,

            * ``"celu"`` -- continuously differentiable ELU
            
              .. math::
                  \\textrm{CELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x / \\alpha} - 1)),

              where the default value for :math:`\\alpha` is 1.0 and can be set
              with ``act_kwargs = {"alpha" : value}``,

            * ``"gelu"`` -- Gaussian Error Linear Unit
              
              .. math::
                  \\textrm{GELU} (x) = x \\Phi (x),

              where :math:`\\Phi (x)` is the cumulative distribution function of
              the unit Gaussian distribution with threshold probability of :math:`x`,

            * ``"silu"`` -- sigmoid linear unit
              
              .. math::
                  \\textrm{SiLU} (x) = x \\textrm{Sigmoid} (x),

            * ``"relu6"`` -- a ReLU activation clamped to 6 for :math:`x \\geq
              6`.

              .. math::
                  \\textrm{ReLU6} = \\textrm{ReLU} (\\textrm{min}(x, 6)),

            * and ``"softmax"`` -- the softmax activation is used to estimate
              the normalised probability of an input with respect to all inputs
              
              .. math::
                  \\textrm{softmax} (x_{i}) = \\frac{e^{x_{i}}}{\\sum_{k} e^{x_{k}}}.
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        ``"kaiming"`` -- learnable parameters are initialised using Kaiming
        initialisation. Options supported are
            * ``"Kaiming"`` or ``"He"`` -- this initialises each parameter as a
              sample for the normal distribution of mean zero and standard
              deviation
              
              .. math::
                  \\sigma = \\frac{g}{\sqrt{n_{l}}},
                  
              where :math:`g` is known as the gain of the nonlinearity and is
              just a number used to scale this standard deviation based on what
              activation function is used. :math:`n_{l}` is either the number of
              input or output connections of the layer, e.g. for a
              fully-connected layer it is the number of input or output nodes.
              The default is to use the number of inputs to a layer but this can
              be changed to using the number of outputs of a layer by using
              ``init_kwargs = {"mode" : "fan_out"}``,

            * ``"Xavier"`` -- this is a special case of He initialisation where
              :math:`n_{l}` is the average number of the input and output nodes,

            * ``"uniform"`` -- initialises the learnable parameters as samples
              of a uniform distribution. This defaults to the unit uniform
              distribution :math:`\\mathcal{U} (0, 1)` but the upper and lower
              bounds can be changed by using ``init_kwargs = {"a" : value1, "b"
              : value2}`` where ``a`` is the lower bound and ``b`` is the upper bound,

            * ``"normal"`` -- initialises the learnable parameters as samples of
              a normal distribution. This defaults to the standard normal
              distribution with mean zero and standard deviation one
              :math:`\\mathcal{N} (0, 1)` but these can be changed via
              ``init_kwargs = {"mean" : value1, "std" : value2}`,

            * ``"constant"`` -- initialises the learnable parameters with the
              same value. There is no default for this value therefore it must
              be specified using ``init_kwargs = {"val" : value}``,

            * ``"eye"`` -- initialises the learnable parameters with the
              identity matrix,

            * ``"kaiming uniform"``/``"he uniform"`` -- similar initialisation
              to Kaiming normal but this time using a uniform distribution
              :math:`\\mathcal{U} (-b, b)` where
              
              .. math::
                  b = g \\sqrt{\\frac{3}{n_{l}}},

            * ``"xavier uniform"`` -- equivalent relationship to He uniform
              initialisation as between the normal cases,

            * ``"orthogonal"`` -- initialises the learnable parameters with an
              orthogonal matrix, i.e. one whose transpose and inverse are equal,

            * ``"sparse"`` -- initialises the learnable parameters with a sparse
              matrix. The fraction of elements in each column to be set to zero
              must be set manually via ``init_kwargs = {"sparsity" : value}``.
              The nonzero values are samples drawn from the normal distribution
              centred on zero with a standard deviation which defaults to 0.01
              but can be set using ``init_kwargs = {"sparsity" : value1, "std" : value2}``

            * or ``None`` -- if no initialisation is specified the default
              PyTorch initialisation is used for the learnable parameters where
              they are samples of the normal distribution centred on zero with a
              standard deviation equal to :math:`1 / \\sqrt{\\textrm{dim} (x)}`.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is ``False``.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    lin_kwargs : dict, optional
        Additional keyword arguments to be passed to the ``torch.nn.Linear``
        module. Default is ``{}`` -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is ``{}``.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is ``{}``.
    init_kwargs : dict, optional
        Additional keyword arguments to be passed to the initialisation being
        used. Default is ``{}``.
    """

    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        lin_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        init_kwargs: Dict = {}
    ) -> None:
        super().__init__()

        self.lin = nn.Linear(in_nodes, out_nodes, bias=bias, **lin_kwargs)

        self.norm = self._assign_normalisation(out_nodes, normalisation=normalisation, norm_kwargs=norm_kwargs)

        self.act = self._assign_activation(activation=activation, act_kwargs=act_kwargs)

        self._init_weights(initialisation, bias, activation, init_kwargs)

        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    @property
    def weight(self):
        return self.lin.weight

    @property
    def bias(self):
        return self.lin.bias

    def _assign_normalisation(self, out_nodes: int, normalisation: Optional[str], norm_kwargs: Dict) -> Optional[nn.Module]:
        if (isinstance(normalisation, str)):
            if normalisation.lower() == "batch":
                return nn.BatchNorm1d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "instance":
                return nn.InstanceNorm1d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "group":
                return nn.GroupNorm(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "layer":
                return nn.LayerNorm(out_nodes, **norm_kwargs)
        else:
            return None
    
    def _assign_activation(self, activation: str, act_kwargs: Dict) -> nn.Module:
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "elu":
            return nn.ELU(inplace=True, **act_kwargs)
        elif activation.lower() == "prelu":
            return nn.PReLU(**act_kwargs)
        elif activation.lower() == "rrelu":
            return nn.RReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "selu":
            return nn.SELU(inplace=True)
        elif activation.lower() == "celu":
            return nn.CELU(inplace=True, **act_kwargs)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation.lower() == "softmax":
            return nn.Softmax(**act_kwargs)
        else:
            raise NotImplementedError("Pester John to add this.")

    def _init_weights(self, initialisation: Optional[str], bias: bool, activation: str, init_kwargs: Dict) -> None:
        if initialisation:
            if initialisation.lower() == "kaiming" or "he":
                nn.init.kaiming_normal_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_normal_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier":
                nn.init.xavier_normal_(
                    self.weight, gain=nn.init.calculate_gain(activation)
                )
                if bias:
                    nn.init.xavier_normal_(
                        self.bias, gain=nn.init.calculate_gain(activation)
                    )
            elif initialisation.lower() == "uniform":
                nn.init.uniform_(self.weight, **init_kwargs)
                if bias:
                    nn.init.uniform_(self.bias, **init_kwargs)
            elif initialisation.lower() == "normal":
                nn.init.normal_(self.weight, **init_kwargs)
                if bias:
                    nn.init.normal_(self.bias, **init_kwargs)
            elif initialisation.lower() == "constant":
                nn.init.constant_(self.weight, **init_kwargs)
                if bias:
                    nn.init.constant_(self.bias, **init_kwargs)
            elif initialisation.lower() == "eye":
                nn.init.eye_(self.weight)
                if bias:
                    nn.init.ones_(self.bias)
            elif initialisation.lower() == "kaiming uniform" or "he uniform":
                nn.init.kaiming_uniform_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_uniform_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier uniform":
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "orthogonal":
                nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "sparse":
                nn.init.sparse_(self.weight, **init_kwargs)
                if bias:
                    nn.init.sparse_(self.bias, **init_kwargs)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of linear,
        normalisation, activation.

        Parameters
        ----------
        inp : torch.Tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.Tensor
             The transformed data.
        """
        out = self.lin(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout:
            out = self.dropout(out)

        return out


class ConvLayer(nn.Module):
    """
    A modifiable convolutional layer for deep networks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the convolutional layer.
    out_channels : int
        The number of output channels of the convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the ``torch.nn.Conv2d``
        linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        ``"reflect"``. Other options available
        `here <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is ``None`` -- no
        normalisation used. Options ``"batch"``, ``"instance"``, ``"group"`` and
        ``"layer"`` are supported to perform batch, instance, group or layer normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is ``"relu"`` -- uses the
        :abbr:`ReLU (rectified linear unit)` nonlinearity. Options supported are
            * ``"relu"`` -- a piecewise function where positive values are
              mapped to themselves and negative values are mapped to 0. This can
              be written as :math:`\\textrm{ReLU}(x) = \\textrm{max}(0, x)`,

            * ``"leaky_relu"`` -- a variant of the ReLU where negative values
              are mapped to :math:`\\alpha x` where :math:`\\alpha < 1`. The
              default value is 0.01 but this gradient can be set using the
              ``act_kwargs`` dictionary by setting ``act_kwargs =
              {"negative_slope" : value}``. Mathematically, this can be written

              .. math:: 
                  \\textrm{Leaky ReLU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha x),

            * ``"sigmoid"`` -- the sigmoid function
              
              .. math::
                  \\textrm{Sigmoid} (x) = \\frac{1}{1 + e^{-x}},

            * ``"tanh"`` -- the hyperbolic tangent function
              
              .. math::
                  \\textrm{tanh} (x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}},

            * ``"elu"`` -- the exponential linear unit is similar to the Leaky
              ReLU but the negative values are mapped to a scaled exponential
              curve
              
              .. math::
                  \\textrm{ELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x} - 1)),
                                      
              where :math:`\\alpha` defaults to 1.0 and can be assigned using
              the ``act_kwargs = {"alpha" : value}``,

            * ``"prelu"`` -- parameterised ReLU. This performs the same
              transformation as the Leaky ReLU but the :math:`\\alpha` parameter
              is learned. The starting value for :math:`\\alpha` is 0.25 but
              this can be changed by doing ``act_kwargs = {"init" : value}``,

            * ``"rrelu"`` -- randomised ReLU. Another variant of the Leaky ReLU
              where the value of :math:`\\alpha` is sampled from the uniform
              distribution :math:`\\mathcal{U} (\\textrm{lower},
              \\textrm{upper})` where the default value for lower is 1/8 and the
              default value for upper is 1/3. These can be set by ``act_kwargs =
              {"lower" : value1, "upper" : value2}``,

            * ``"selu"`` -- scaled ELU. Performs the ELU transformation
              multiplied by a fixed scaling factor
              
              .. math::
                  \\textrm{SELU} (x) = s * \\textrm{ELU} (x; \\alpha),
                  
              where :math:`s` = 1.0507009873554804934193349852946 and
              :math:`\\alpha` = 1.6732632423543772848170429916717,

            * ``"celu"`` -- continuously differentiable ELU
            
              .. math::
                  \\textrm{CELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x / \\alpha} - 1)),

              where the default value for :math:`\\alpha` is 1.0 and can be set
              with ``act_kwargs = {"alpha" : value}``,

            * ``"gelu"`` -- Gaussian Error Linear Unit
              
              .. math::
                  \\textrm{GELU} (x) = x \\Phi (x),

              where :math:`\\Phi (x)` is the cumulative distribution function of
              the unit Gaussian distribution with threshold probability of :math:`x`,

            * ``"silu"`` -- sigmoid linear unit
              
              .. math::
                  \\textrm{SiLU} (x) = x \\textrm{Sigmoid} (x),

            * ``"relu6"`` -- a ReLU activation clamped to 6 for :math:`x \\geq
              6`.

              .. math::
                  \\textrm{ReLU6} = \\textrm{ReLU} (\\textrm{min}(x, 6)),

            * and ``"softmax"`` -- the softmax activation is used to estimate
              the normalised probability of an input with respect to all inputs
              
              .. math::
                  \\textrm{softmax} (x_{i}) = \\frac{e^{x_{i}}}{\\sum_{k} e^{x_{k}}}.
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        ``"kaiming"`` -- learnable parameters are initialised using Kaiming
        initialisation. Options supported are
            * ``"Kaiming"`` or ``"He"`` -- this initialises each parameter as a
              sample for the normal distribution of mean zero and standard
              deviation
              
              .. math::
                  \\sigma = \\frac{g}{\sqrt{n_{l}}},
                  
              where :math:`g` is known as the gain of the nonlinearity and is
              just a number used to scale this standard deviation based on what
              activation function is used. :math:`n_{l}` is either the number of
              input or output connections of the layer, e.g. for a
              convolutional layer it is the number of input or output channels/feature maps.
              The default is to use the number of inputs to a layer but this can
              be changed to using the number of outputs of a layer by using
              ``init_kwargs = {"mode" : "fan_out"}``,

            * ``"Xavier"`` -- this is a special case of He initialisation where
              :math:`n_{l}` is the average number of the input and output nodes,

            * ``"uniform"`` -- initialises the learnable parameters as samples
              of a uniform distribution. This defaults to the unit uniform
              distribution :math:`\\mathcal{U} (0, 1)` but the upper and lower
              bounds can be changed by using ``init_kwargs = {"a" : value1, "b"
              : value2}`` where ``a`` is the lower bound and ``b`` is the upper bound,

            * ``"normal"`` -- initialises the learnable parameters as samples of
              a normal distribution. This defaults to the standard normal
              distribution with mean zero and standard deviation one
              :math:`\\mathcal{N} (0, 1)` but these can be changed via
              ``init_kwargs = {"mean" : value1, "std" : value2}`,

            * ``"constant"`` -- initialises the learnable parameters with the
              same value. There is no default for this value therefore it must
              be specified using ``init_kwargs = {"val" : value}``,

            * ``"kaiming uniform"``/``"he uniform"`` -- similar initialisation
              to Kaiming normal but this time using a uniform distribution
              :math:`\\mathcal{U} (-b, b)` where
              
              .. math::
                  b = g \\sqrt{\\frac{3}{n_{l}}},

            * ``"xavier uniform"`` -- equivalent relationship to He uniform
              initialisation as between the normal cases,

            * ``"orthogonal"`` -- initialises the learnable parameters with an
              orthogonal matrix, i.e. one whose transpose and inverse are equal,

            * ``"sparse"`` -- initialises the learnable parameters with a sparse
              matrix. The fraction of elements in each column to be set to zero
              must be set manually via ``init_kwargs = {"sparsity" : value}``.
              The nonzero values are samples drawn from the normal distribution
              centred on zero with a standard deviation which defaults to 0.01
              but can be set using ``init_kwargs = {"sparsity" : value1, "std" : value2}``

            * or ``None`` -- if no initialisation is specified the default
              PyTorch initialisation is used for the learnable parameters where
              they are samples of the normal distribution centred on zero with a
              standard deviation equal to :math:`1 / \\sqrt{\\textrm{dim} (x)}`.
    upsample : bool, optional
        Whether or not the convolutional layer will be used to spatially
        upsample the data using a linear interpolation. Default is ``False``.
    upsample_factor : int, optional
        The factor that the spatial dimensions is upsampled. Default is 2.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is ``False``.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the ``torch.nn.Conv2d``
        module. Default is ``{}`` -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is ``{}``.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is ``{}``.
    init_kwargs : dict, optional
        Additional keyword arguments to be passed to the initialisation being
        used. Default is ``{}``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        upsample: bool = False,
        upsample_factor: int = 2,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        init_kwargs: Dict = {}
    ) -> None:
        super(ConvLayer, self).__init__()

        self.upsample = upsample
        self.upsample_factor = upsample_factor

        if isinstance(kernel, int):
            padding = (kernel - 1) // 2
        else:
            padding = [(x - 1) // 2 for x in kernel]

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )

        self.norm = self._assign_normalisation(out_channels, normalisation=normalisation, norm_kwargs=norm_kwargs)

        self.act = self._assign_activation(activation=activation, act_kwargs=act_kwargs)

        self._init_weights(initialisation, bias, activation, init_kwargs)

        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def _assign_normalisation(self, out_nodes: int, normalisation: Optional[str], norm_kwargs: Dict) -> Optional[nn.Module]:
        if (isinstance(normalisation, str)):
            if normalisation.lower() == "batch":
                return nn.BatchNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "instance":
                return nn.InstanceNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "group":
                return nn.GroupNorm(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "layer":
                return nn.LayerNorm(out_nodes, **norm_kwargs)
        else:
            return None
    
    def _assign_activation(self, activation: str, act_kwargs: Dict) -> nn.Module:
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "elu":
            return nn.ELU(inplace=True, **act_kwargs)
        elif activation.lower() == "prelu":
            return nn.PReLU(**act_kwargs)
        elif activation.lower() == "rrelu":
            return nn.RReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "selu":
            return nn.SELU(inplace=True)
        elif activation.lower() == "celu":
            return nn.CELU(inplace=True, **act_kwargs)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation.lower() == "softmax":
            return nn.Softmax(**act_kwargs)
        else:
            raise NotImplementedError("Pester John to add this.")

    def _init_weights(self, initialisation: Optional[str], bias: bool, activation: str, init_kwargs: Dict) -> None:
        if initialisation:
            if initialisation.lower() == "kaiming" or "he":
                nn.init.kaiming_normal_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_normal_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier":
                nn.init.xavier_normal_(
                    self.weight, gain=nn.init.calculate_gain(activation)
                )
                if bias:
                    nn.init.xavier_normal_(
                        self.bias, gain=nn.init.calculate_gain(activation)
                    )
            elif initialisation.lower() == "uniform":
                nn.init.uniform_(self.weight, **init_kwargs)
                if bias:
                    nn.init.uniform_(self.bias, **init_kwargs)
            elif initialisation.lower() == "normal":
                nn.init.normal_(self.weight, **init_kwargs)
                if bias:
                    nn.init.normal_(self.bias, **init_kwargs)
            elif initialisation.lower() == "constant":
                nn.init.constant_(self.weight, **init_kwargs)
                if bias:
                    nn.init.constant_(self.bias, **init_kwargs)
            elif initialisation.lower() == "kaiming uniform" or "he uniform":
                nn.init.kaiming_uniform_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_uniform_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier uniform":
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "orthogonal":
                nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "sparse":
                nn.init.sparse_(self.weight, **init_kwargs)
                if bias:
                    nn.init.sparse_(self.bias, **init_kwargs)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of upsample,
        linear, normalisation, activation, dropout if selected.

        Parameters
        ----------
        inp : torch.Tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.Tensor
             The transformed data.
        """
        if self.upsample:
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)
        out = self.conv(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out


class ConvTranspLayer(nn.Module):
    """
    A modifiable transpose convolutional layer.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the transpose convolutional layer.
    out_channels : int
        The number of output channels of the transpose convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the
        ``torch.nn.Conv2dTranspose`` linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1. This is also used
        to define the ``output_padding`` kwarg for the
        ``torch.nn.ConvTranspose2d`` module. ``output_padding`` provides
        implicit padding on the output of the transpose convolution when
        ``stride > 1`` to deterministically find the correct output shape. For
        more information, please see `here <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        ``"reflect"``. Other options available
        `here <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2dTranspose>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is ``None`` -- no
        normalisation used. Options ``"batch"``, ``"instance"``, ``"group"`` and
        ``"layer"`` are supported to perform batch, instance, group or layer normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is ``"relu"`` -- uses the
        :abbr:`ReLU (rectified linear unit)` nonlinearity. Options supported are
            * ``"relu"`` -- a piecewise function where positive values are
              mapped to themselves and negative values are mapped to 0. This can
              be written as :math:`\\textrm{ReLU}(x) = \\textrm{max}(0, x)`,

            * ``"leaky_relu"`` -- a variant of the ReLU where negative values
              are mapped to :math:`\\alpha x` where :math:`\\alpha < 1`. The
              default value is 0.01 but this gradient can be set using the
              ``act_kwargs`` dictionary by setting ``act_kwargs =
              {"negative_slope" : value}``. Mathematically, this can be written

              .. math:: 
                  \\textrm{Leaky ReLU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha x),

            * ``"sigmoid"`` -- the sigmoid function
              
              .. math::
                  \\textrm{Sigmoid} (x) = \\frac{1}{1 + e^{-x}},

            * ``"tanh"`` -- the hyperbolic tangent function
              
              .. math::
                  \\textrm{tanh} (x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}},

            * ``"elu"`` -- the exponential linear unit is similar to the Leaky
              ReLU but the negative values are mapped to a scaled exponential
              curve
              
              .. math::
                  \\textrm{ELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x} - 1)),
                                      
              where :math:`\\alpha` defaults to 1.0 and can be assigned using
              the ``act_kwargs = {"alpha" : value}``,

            * ``"prelu"`` -- parameterised ReLU. This performs the same
              transformation as the Leaky ReLU but the :math:`\\alpha` parameter
              is learned. The starting value for :math:`\\alpha` is 0.25 but
              this can be changed by doing ``act_kwargs = {"init" : value}``,

            * ``"rrelu"`` -- randomised ReLU. Another variant of the Leaky ReLU
              where the value of :math:`\\alpha` is sampled from the uniform
              distribution :math:`\\mathcal{U} (\\textrm{lower},
              \\textrm{upper})` where the default value for lower is 1/8 and the
              default value for upper is 1/3. These can be set by ``act_kwargs =
              {"lower" : value1, "upper" : value2}``,

            * ``"selu"`` -- scaled ELU. Performs the ELU transformation
              multiplied by a fixed scaling factor
              
              .. math::
                  \\textrm{SELU} (x) = s * \\textrm{ELU} (x; \\alpha),
                  
              where :math:`s` = 1.0507009873554804934193349852946 and
              :math:`\\alpha` = 1.6732632423543772848170429916717,

            * ``"celu"`` -- continuously differentiable ELU
            
              .. math::
                  \\textrm{CELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x / \\alpha} - 1)),

              where the default value for :math:`\\alpha` is 1.0 and can be set
              with ``act_kwargs = {"alpha" : value}``,

            * ``"gelu"`` -- Gaussian Error Linear Unit
              
              .. math::
                  \\textrm{GELU} (x) = x \\Phi (x),

              where :math:`\\Phi (x)` is the cumulative distribution function of
              the unit Gaussian distribution with threshold probability of :math:`x`,

            * ``"silu"`` -- sigmoid linear unit
              
              .. math::
                  \\textrm{SiLU} (x) = x \\textrm{Sigmoid} (x),

            * ``"relu6"`` -- a ReLU activation clamped to 6 for :math:`x \\geq
              6`.

              .. math::
                  \\textrm{ReLU6} = \\textrm{ReLU} (\\textrm{min}(x, 6)),

            * and ``"softmax"`` -- the softmax activation is used to estimate
              the normalised probability of an input with respect to all inputs
              
              .. math::
                  \\textrm{softmax} (x_{i}) = \\frac{e^{x_{i}}}{\\sum_{k} e^{x_{k}}}.
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        ``"kaiming"`` -- learnable parameters are initialised using Kaiming
        initialisation. Options supported are
            * ``"Kaiming"`` or ``"He"`` -- this initialises each parameter as a
              sample for the normal distribution of mean zero and standard
              deviation
              
              .. math::
                  \\sigma = \\frac{g}{\sqrt{n_{l}}},
                  
              where :math:`g` is known as the gain of the nonlinearity and is
              just a number used to scale this standard deviation based on what
              activation function is used. :math:`n_{l}` is either the number of
              input or output connections of the layer, e.g. for a
              convolutional layer it is the number of input or output channels/feature maps.
              The default is to use the number of inputs to a layer but this can
              be changed to using the number of outputs of a layer by using
              ``init_kwargs = {"mode" : "fan_out"}``,

            * ``"Xavier"`` -- this is a special case of He initialisation where
              :math:`n_{l}` is the average number of the input and output nodes,

            * ``"uniform"`` -- initialises the learnable parameters as samples
              of a uniform distribution. This defaults to the unit uniform
              distribution :math:`\\mathcal{U} (0, 1)` but the upper and lower
              bounds can be changed by using ``init_kwargs = {"a" : value1, "b"
              : value2}`` where ``a`` is the lower bound and ``b`` is the upper bound,

            * ``"normal"`` -- initialises the learnable parameters as samples of
              a normal distribution. This defaults to the standard normal
              distribution with mean zero and standard deviation one
              :math:`\\mathcal{N} (0, 1)` but these can be changed via
              ``init_kwargs = {"mean" : value1, "std" : value2}`,

            * ``"constant"`` -- initialises the learnable parameters with the
              same value. There is no default for this value therefore it must
              be specified using ``init_kwargs = {"val" : value}``,

            * ``"kaiming uniform"``/``"he uniform"`` -- similar initialisation
              to Kaiming normal but this time using a uniform distribution
              :math:`\\mathcal{U} (-b, b)` where
              
              .. math::
                  b = g \\sqrt{\\frac{3}{n_{l}}},

            * ``"xavier uniform"`` -- equivalent relationship to He uniform
              initialisation as between the normal cases,

            * ``"orthogonal"`` -- initialises the learnable parameters with an
              orthogonal matrix, i.e. one whose transpose and inverse are equal,

            * ``"sparse"`` -- initialises the learnable parameters with a sparse
              matrix. The fraction of elements in each column to be set to zero
              must be set manually via ``init_kwargs = {"sparsity" : value}``.
              The nonzero values are samples drawn from the normal distribution
              centred on zero with a standard deviation which defaults to 0.01
              but can be set using ``init_kwargs = {"sparsity" : value1, "std" : value2}``

            * or ``None`` -- if no initialisation is specified the default
              PyTorch initialisation is used for the learnable parameters where
              they are samples of the normal distribution centred on zero with a
              standard deviation equal to :math:`1 / \\sqrt{\\textrm{dim} (x)}`.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is ``False``.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the ``torch.nn.Conv2d``
        module. Default is ``{}`` -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is ``{}``.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is ``{}``.
    init_kwargs : dict, optional
        Additional keyword arguments to be passed to the initialisation being
        used. Default is ``{}``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        init_kwargs: Dict = {}
    ) -> None:
        super(ConvTranspLayer, self).__init__()

        if isinstance(kernel, int):
            padding = (kernel - 1) // 2
        else:
            padding = [(x - 1) // 2 for x in kernel]

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            output_padding=stride // 2,
            **conv_kwargs
        )

        self.norm = self._assign_normalisation(out_channels, normalisation=normalisation, norm_kwargs=norm_kwargs)

        self.act = self._assign_activation(activation=activation, act_kwargs=act_kwargs)

        self._init_weights(initialisation, bias, activation, init_kwargs)

        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def _assign_normalisation(self, out_nodes: int, normalisation: Optional[str], norm_kwargs: Dict) -> Optional[nn.Module]:
        if (isinstance(normalisation, str)):
            if normalisation.lower() == "batch":
                return nn.BatchNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "instance":
                return nn.InstanceNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "group":
                return nn.GroupNorm(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "layer":
                return nn.LayerNorm(out_nodes, **norm_kwargs)
        else:
            return None
    
    def _assign_activation(self, activation: str, act_kwargs: Dict) -> nn.Module:
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "elu":
            return nn.ELU(inplace=True, **act_kwargs)
        elif activation.lower() == "prelu":
            return nn.PReLU(**act_kwargs)
        elif activation.lower() == "rrelu":
            return nn.RReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "selu":
            return nn.SELU(inplace=True)
        elif activation.lower() == "celu":
            return nn.CELU(inplace=True, **act_kwargs)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation.lower() == "softmax":
            return nn.Softmax(**act_kwargs)
        else:
            raise NotImplementedError("Pester John to add this.")

    def _init_weights(self, initialisation: Optional[str], bias: bool, activation: str, init_kwargs: Dict) -> None:
        if initialisation:
            if initialisation.lower() == "kaiming" or "he":
                nn.init.kaiming_normal_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_normal_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier":
                nn.init.xavier_normal_(
                    self.weight, gain=nn.init.calculate_gain(activation)
                )
                if bias:
                    nn.init.xavier_normal_(
                        self.bias, gain=nn.init.calculate_gain(activation)
                    )
            elif initialisation.lower() == "uniform":
                nn.init.uniform_(self.weight, **init_kwargs)
                if bias:
                    nn.init.uniform_(self.bias, **init_kwargs)
            elif initialisation.lower() == "normal":
                nn.init.normal_(self.weight, **init_kwargs)
                if bias:
                    nn.init.normal_(self.bias, **init_kwargs)
            elif initialisation.lower() == "constant":
                nn.init.constant_(self.weight, **init_kwargs)
                if bias:
                    nn.init.constant_(self.bias, **init_kwargs)
            elif initialisation.lower() == "kaiming uniform" or "he uniform":
                nn.init.kaiming_uniform_(self.weight, nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_uniform_(self.bias, nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier uniform":
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "orthogonal":
                nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "sparse":
                nn.init.sparse_(self.weight, **init_kwargs)
                if bias:
                    nn.init.sparse_(self.bias, **init_kwargs)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of
        linear, normalisation, activation.

        Parameters
        ----------
        inp : torch.Tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.Tensor
             The transformed data.
        """
        out = self.conv(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out


class ResLayer(nn.Module):
    """
    A modifiable residual layer for deep neural networks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the convolutional layer.
    out_channels : int
        The number of output channels of the convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the ``torch.nn.Conv2d``
        linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        ``"reflect"``. Other options available
        `here <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is ``None`` -- no
        normalisation used. Options ``"batch"``, ``"instance"``, ``"group"`` and
        ``"layer"`` are supported to perform batch, instance, group or layer normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is ``"relu"`` -- uses the
        :abbr:`ReLU (rectified linear unit)` nonlinearity. Options supported are
            * ``"relu"`` -- a piecewise function where positive values are
              mapped to themselves and negative values are mapped to 0. This can
              be written as :math:`\\textrm{ReLU}(x) = \\textrm{max}(0, x)`,

            * ``"leaky_relu"`` -- a variant of the ReLU where negative values
              are mapped to :math:`\\alpha x` where :math:`\\alpha < 1`. The
              default value is 0.01 but this gradient can be set using the
              ``act_kwargs`` dictionary by setting ``act_kwargs =
              {"negative_slope" : value}``. Mathematically, this can be written

              .. math:: 
                  \\textrm{Leaky ReLU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha x),

            * ``"sigmoid"`` -- the sigmoid function
              
              .. math::
                  \\textrm{Sigmoid} (x) = \\frac{1}{1 + e^{-x}},

            * ``"tanh"`` -- the hyperbolic tangent function
              
              .. math::
                  \\textrm{tanh} (x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}},

            * ``"elu"`` -- the exponential linear unit is similar to the Leaky
              ReLU but the negative values are mapped to a scaled exponential
              curve
              
              .. math::
                  \\textrm{ELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x} - 1)),
                                      
              where :math:`\\alpha` defaults to 1.0 and can be assigned using
              the ``act_kwargs = {"alpha" : value}``,

            * ``"prelu"`` -- parameterised ReLU. This performs the same
              transformation as the Leaky ReLU but the :math:`\\alpha` parameter
              is learned. The starting value for :math:`\\alpha` is 0.25 but
              this can be changed by doing ``act_kwargs = {"init" : value}``,

            * ``"rrelu"`` -- randomised ReLU. Another variant of the Leaky ReLU
              where the value of :math:`\\alpha` is sampled from the uniform
              distribution :math:`\\mathcal{U} (\\textrm{lower},
              \\textrm{upper})` where the default value for lower is 1/8 and the
              default value for upper is 1/3. These can be set by ``act_kwargs =
              {"lower" : value1, "upper" : value2}``,

            * ``"selu"`` -- scaled ELU. Performs the ELU transformation
              multiplied by a fixed scaling factor
              
              .. math::
                  \\textrm{SELU} (x) = s * \\textrm{ELU} (x; \\alpha),
                  
              where :math:`s` = 1.0507009873554804934193349852946 and
              :math:`\\alpha` = 1.6732632423543772848170429916717,

            * ``"celu"`` -- continuously differentiable ELU
            
              .. math::
                  \\textrm{CELU} (x; \\alpha) = \\textrm{max} (0, x) + \\textrm{min} (0, \\alpha (e^{x / \\alpha} - 1)),

              where the default value for :math:`\\alpha` is 1.0 and can be set
              with ``act_kwargs = {"alpha" : value}``,

            * ``"gelu"`` -- Gaussian Error Linear Unit
              
              .. math::
                  \\textrm{GELU} (x) = x \\Phi (x),

              where :math:`\\Phi (x)` is the cumulative distribution function of
              the unit Gaussian distribution with threshold probability of :math:`x`,

            * ``"silu"`` -- sigmoid linear unit
              
              .. math::
                  \\textrm{SiLU} (x) = x \\textrm{Sigmoid} (x),

            * ``"relu6"`` -- a ReLU activation clamped to 6 for :math:`x \\geq
              6`.

              .. math::
                  \\textrm{ReLU6} = \\textrm{ReLU} (\\textrm{min}(x, 6)),

            * and ``"softmax"`` -- the softmax activation is used to estimate
              the normalised probability of an input with respect to all inputs
              
              .. math::
                  \\textrm{softmax} (x_{i}) = \\frac{e^{x_{i}}}{\\sum_{k} e^{x_{k}}}.
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        ``"kaiming"`` -- learnable parameters are initialised using Kaiming
        initialisation. Options supported are
            * ``"Kaiming"`` or ``"He"`` -- this initialises each parameter as a
              sample for the normal distribution of mean zero and standard
              deviation
              
              .. math::
                  \\sigma = \\frac{g}{\sqrt{n_{l}}},
                  
              where :math:`g` is known as the gain of the nonlinearity and is
              just a number used to scale this standard deviation based on what
              activation function is used. :math:`n_{l}` is either the number of
              input or output connections of the layer, e.g. for a
              convolutional layer it is the number of input or output channels/feature maps.
              The default is to use the number of inputs to a layer but this can
              be changed to using the number of outputs of a layer by using
              ``init_kwargs = {"mode" : "fan_out"}``,

            * ``"Xavier"`` -- this is a special case of He initialisation where
              :math:`n_{l}` is the average number of the input and output nodes,

            * ``"uniform"`` -- initialises the learnable parameters as samples
              of a uniform distribution. This defaults to the unit uniform
              distribution :math:`\\mathcal{U} (0, 1)` but the upper and lower
              bounds can be changed by using ``init_kwargs = {"a" : value1, "b"
              : value2}`` where ``a`` is the lower bound and ``b`` is the upper bound,

            * ``"normal"`` -- initialises the learnable parameters as samples of
              a normal distribution. This defaults to the standard normal
              distribution with mean zero and standard deviation one
              :math:`\\mathcal{N} (0, 1)` but these can be changed via
              ``init_kwargs = {"mean" : value1, "std" : value2}`,

            * ``"constant"`` -- initialises the learnable parameters with the
              same value. There is no default for this value therefore it must
              be specified using ``init_kwargs = {"val" : value}``,

            * ``"kaiming uniform"``/``"he uniform"`` -- similar initialisation
              to Kaiming normal but this time using a uniform distribution
              :math:`\\mathcal{U} (-b, b)` where
              
              .. math::
                  b = g \\sqrt{\\frac{3}{n_{l}}},

            * ``"xavier uniform"`` -- equivalent relationship to He uniform
              initialisation as between the normal cases,

            * ``"orthogonal"`` -- initialises the learnable parameters with an
              orthogonal matrix, i.e. one whose transpose and inverse are equal,

            * ``"sparse"`` -- initialises the learnable parameters with a sparse
              matrix. The fraction of elements in each column to be set to zero
              must be set manually via ``init_kwargs = {"sparsity" : value}``.
              The nonzero values are samples drawn from the normal distribution
              centred on zero with a standard deviation which defaults to 0.01
              but can be set using ``init_kwargs = {"sparsity" : value1, "std" : value2}``

            * or ``None`` -- if no initialisation is specified the default
              PyTorch initialisation is used for the learnable parameters where
              they are samples of the normal distribution centred on zero with a
              standard deviation equal to :math:`1 / \\sqrt{\\textrm{dim} (x)}`.
    upsample : bool, optional
        Whether or not the convolutional layer will be used to spatially
        upsample the data using a linear interpolation. Default is ``False``.
    upsample_factor : int, optional
        The factor that the spatial dimensions is upsampled. Default is 2.
    use_dropout : bool, optional
        Whether or not to apply dropout after the first activation. Default is ``False``.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the ``torch.nn.Conv2d``
        module. Default is ``{}`` -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is ``{}``.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is ``{}``.
    init_kwargs : dict, optional
        Additional keyword arguments to be passed to the initialisation being
        used. Default is ``{}``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        upsample: bool = False,
        upsample_factor: int = 2,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        init_kwargs: Dict = {}
    ) -> None:
        super(ResLayer, self).__init__()

        self.upsample = upsample
        self.upsample_factor = upsample_factor

        if isinstance(kernel, int):
            padding = (kernel - 1) // 2
        else:
            padding = [(x - 1) // 2 for x in kernel]

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )

        self.norm1 = self._assign_normalisation(out_channels, normalisation=normalisation, norm_kwargs=norm_kwargs)
        self.norm2 = self._assign_normalisation(out_channels, normalisation=normalisation, norm_kwargs=norm_kwargs)

        self.act = self._assign_activation(activation=activation, act_kwargs=act_kwargs)

        self._init_weights(initialisation, bias, activation, init_kwargs)

        # if the number of channels is changing and there is not an upsample then self.downsample is needed to transform the identity of the residual layer to the dimensions of the output so they can be added
        # if the number of channels is changing and there is also upsampling then self.downsample is needed to transform the number of channels of the identity of the residual layer with the upscaling of the identity taking place elsewhere
        # if the number of channels stays the same then the 1x1 convolution is not needed
        if in_channels != out_channels and not upsample:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        elif in_channels != out_channels and upsample:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
        else:
            self.downsample = None

        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    @property
    def weight(self):
        return self.conv1.weight, self.conv2.weight

    @property
    def bias(self):
        return self.conv1.bias, self.conv2.bias

    def _assign_normalisation(self, out_nodes: int, normalisation: Optional[str], norm_kwargs: Dict) -> Optional[nn.Module]:
        if (isinstance(normalisation, str)):
            if normalisation.lower() == "batch":
                return nn.BatchNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "instance":
                return nn.InstanceNorm2d(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "group":
                return nn.GroupNorm(out_nodes, **norm_kwargs)
            elif normalisation.lower() == "layer":
                return nn.LayerNorm(out_nodes, **norm_kwargs)
        else:
            return None
    
    def _assign_activation(self, activation: str, act_kwargs: Dict) -> nn.Module:
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "elu":
            return nn.ELU(inplace=True, **act_kwargs)
        elif activation.lower() == "prelu":
            return nn.PReLU(**act_kwargs)
        elif activation.lower() == "rrelu":
            return nn.RReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "selu":
            return nn.SELU(inplace=True)
        elif activation.lower() == "celu":
            return nn.CELU(inplace=True, **act_kwargs)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation.lower() == "softmax":
            return nn.Softmax(**act_kwargs)
        else:
            raise NotImplementedError("Pester John to add this.")

    def _init_weights(self, initialisation: Optional[str], bias: bool, activation: str, init_kwargs: Dict) -> None:
        if initialisation:
            if initialisation.lower() == "kaiming" or "he":
                nn.init.kaiming_normal_(self.weight[0], nonlinearity=activation, **init_kwargs)
                nn.init.kaiming_normal_(self.weight[1], nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_normal_(self.bias[0], nonlinearity=activation, **init_kwargs)
                    nn.init.kaiming_normal_(self.bias[1], nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier":
                nn.init.xavier_normal_(
                    self.weight[0], gain=nn.init.calculate_gain(activation)
                )
                nn.init.xavier_normal_(
                    self.weight[1], gain=nn.init.calculate_gain(activation)
                )
                if bias:
                    nn.init.xavier_normal_(
                        self.bias[0], gain=nn.init.calculate_gain(activation)
                    )
                    nn.init.xavier_normal_(
                        self.bias[1], gain=nn.init.calculate_gain(activation)
                    )
            elif initialisation.lower() == "uniform":
                nn.init.uniform_(self.weight[0], **init_kwargs)
                nn.init.uniform_(self.weight[1], **init_kwargs)
                if bias:
                    nn.init.uniform_(self.bias[0], **init_kwargs)
                    nn.init.uniform_(self.bias[1], **init_kwargs)
            elif initialisation.lower() == "normal":
                nn.init.normal_(self.weight[0], **init_kwargs)
                nn.init.normal_(self.weight[1], **init_kwargs)
                if bias:
                    nn.init.normal_(self.bias[0], **init_kwargs)
                    nn.init.normal_(self.bias[1], **init_kwargs)
            elif initialisation.lower() == "constant":
                nn.init.constant_(self.weight[0], **init_kwargs)
                nn.init.constant_(self.weight[1], **init_kwargs)
                if bias:
                    nn.init.constant_(self.bias[0], **init_kwargs)
                    nn.init.constant_(self.bias[1], **init_kwargs)
            elif initialisation.lower() == "kaiming uniform" or "he uniform":
                nn.init.kaiming_uniform_(self.weight[0], nonlinearity=activation, **init_kwargs)
                nn.init.kaiming_uniform_(self.weight[1], nonlinearity=activation, **init_kwargs)
                if bias:
                    nn.init.kaiming_uniform_(self.bias[0], nonlinearity=activation, **init_kwargs)
                    nn.init.kaiming_uniform_(self.bias[1], nonlinearity=activation, **init_kwargs)
            elif initialisation.lower() == "xavier uniform":
                nn.init.xavier_uniform_(self.weight[0], gain=nn.init.calculate_gain(activation))
                nn.init.xavier_uniform_(self.weight[1], gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.xavier_uniform_(self.bias[0], gain=nn.init.calculate_gain(activation))
                    nn.init.xavier_uniform_(self.bias[1], gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "orthogonal":
                nn.init.orthogonal_(self.weight[0], gain=nn.init.calculate_gain(activation))
                nn.init.orthogonal_(self.weight[1], gain=nn.init.calculate_gain(activation))
                if bias:
                    nn.init.orthogonal_(self.bias[0], gain=nn.init.calculate_gain(activation))
                    nn.init.orthogonal_(self.bias[1], gain=nn.init.calculate_gain(activation))
            elif initialisation.lower() == "sparse":
                nn.init.sparse_(self.weight[0], **init_kwargs)
                nn.init.sparse_(self.weight[1], **init_kwargs)
                if bias:
                    nn.init.sparse_(self.bias[0], **init_kwargs)
                    nn.init.sparse_(self.bias[1], **init_kwargs)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of
        linear, normalisation, activation, dropout (if selected), linear
        normalisation, adding the input, activation.

        Parameters
        ----------
        inp : torch.Tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.Tensor
             The transformed data.
        """
        identity = inp.clone()

        if self.upsample:
            identity = F.interpolate(identity, scale_factor=self.upsample_factor)
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)

        out = self.conv1(inp)
        if self.norm1:
            out = self.norm1(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)

        return out
