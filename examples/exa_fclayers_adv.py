"""
.. _fclayer_adv_exa:

Advanced Usage of the Fully-connected Layer
===========================================
"""

# %%
# The following makes the assumption that you are already comfortable with the
# contents of :ref:`fclayer_exa`.
#
# As always, the layer must be imported

from cmacpy.nn.neural_network_layers import FCLayer

# %%
# Normalisation
# -------------
# A third, often vital, operation part of any neural network layer is the
# normalisation layer. Normalisation across batches of data (known as "batch
# normalisation") is useful in reducing the dynamic range of your batches at the
# expense of having a learnable mean and standard deviation. In the grand scheme
# of having millions of fitted parameters in your network, having a couple more
# due to batch normalisation is preferable over the network taking some time to
# learn the data distribution.
#
# Normalisation can be added to an ``FCLayer`` object using the keyword argument
# ``normalisation``

fclayer = FCLayer(3, 5, normalisation="batch", initialisation=None)

# %%
# Valid options here are ``"batch"`` and ``"instance"`` to add batch
# normalisation and instance normalisation respectively but batch normalisation
# is typically the most useful. The layout of the layer is then

print(fclayer)

# %%
# The normalisation operation itself can be accessed via the ``.norm`` attribute
# of the class

print(fclayer.norm)

# %%
# Dropout
# -------
# Another technique that can be added to neural network layers is called
# "dropout". Dropout will assign a probability to each of the connections in a
# fully-connected layer and randomly not use those transformations during
# learning iterations. This is employed in deeper networks to avoid overfitting
# with the intuition that the model only being able to use some of its
# parameters while training will result in a more general model. An example of
# this is shown in the image below: using our same fully-connected layer defined
# above, there is a probability that an input won't be used to calculate an
# output. Instead the output will be estimated from a subset of the parameters.
# This is indicated by the dashed lines e.g. without dropout the output
# :math:`y_{1}` is calculated by combining the inputs :math:`x_{1}, x_{2},
# x_{3}`, however in this example the connection from :math:`x_{2}` to
# :math:`y_{1}`
# is dropped meaning :math:`y_{1}` is now calculated from just :math:`x_{1},
# x_{3}`
# 
# .. image:: ../images/fclayer_withdropout.png
#     :width: 400
#     :align: center
# 
# There are two keyword arguments associated with dropout here: ``use_dropout``
# which can be set to ``True`` if the user wishes to include dropout in the
# layer and ``dropout_prob`` which is the probability assigned to each
# connection of whether or not it will be dropped (the default value for this is
# 0.5, 50% chance of the connection not being used).
#
# For example if we wanted to make an ``FCLayer`` which uses dropout and each
# connection has a 30% chance of being dropped this would be formulated like so

fclayer = FCLayer(
    3, 5, normalisation="batch", initialisation=None, use_dropout=True, dropout_prob=0.3
)

print(fclayer)

# %%
# The dropout operation can then be accessed via the ``.dropout`` attribute of
# the class.

print(fclayer.dropout)

# %%
# Initialisation
# --------------
# The last thing that can be added to the ``FCLayer`` is a different
# initialisation scheme for the weights. So far, we have been setting the
# ``initalisation`` keyword argument to ``None`` which causes the learnable
# parameters to be initialised using the standard method discussed in
# :ref:`fclayer_exa`. Other initialisation methods can be employed through this
# kwarg, namely He initialisation and Xavier initialisation.
#
# He initialisation (the default), was used for `the first deep learning
# algorithm that surpassed human level classification percentage
# <https://arxiv.org/abs/1502.01852>`_, bases the initialisation on drawing
# random samples from a normal distribution with mean zero and standard
# deviation inversely proportional to the number of connections in a layer and
# proportional to a value describing the effect the non-linear function has on
# the variance of the output distribution (for a mathematical derivation of this
# see the paper above) and using these as the starting points for the weights.
#
# Xavier initialisation is a special case of He initialisation where it is
# assumed that the non-linearity does not contribute to the variance of the
# output distribution of the layer.
#
# Initialisation using both schemes is shown below

fclayer = FCLayer(3, 5, normalisation="batch", initialisation="he")

print(fclayer.weight)

fclayer = FCLayer(3, 5, normalisation="batch", initialisation="xavier")

print(fclayer.weight)

# sphinx_gallery_thumbnail_path = "./images/fclayer_withdropout.png"