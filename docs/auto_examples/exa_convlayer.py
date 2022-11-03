"""
.. _convlayer_exa:

Using the Convolutional Layer
=============================
Convolutions are an important tool in image processing and a key part to using
deep learning for image analysis. For a *very* good intuitive example on how
convolutions work see `this blog post
<https://betterexplained.com/articles/intuitive-convolution/>`_. For now, we
will focus on the two dimensional convolution between an image and a kernel.
Realistically, an image is just a matrix of numbers. The job of the kernel is to
highlight relevant information in the image with the resulting convolution
quantifying the result of overlapping areas of the image with the kernel. This
is shown pictorially below.

.. image:: ../images/conv.png
    :width: 600
    :align: center

The image is the :math:`4 \\times 4` array and the kernel is the :math:`2
\\times 2` array. Each result from the convolution is the weighted sum of the
kernel and a region of the image e.g. following the example above

.. math::

    y_{1} &= \\theta_{1} x_{1} + \\theta_{2} x_{2} + \\theta_{3} x_{5} + \\theta_{4} x_{6}

    y_{2} &= \\theta_{1} x_{2} + \\theta_{2} x_{3} + \\theta_{3} x_{6} +
    \\theta_{4} x_{7}

    &\\vdots

    y_{9} &= \\theta_{1} x_{11} + \\theta_{2} x_{12} + \\theta_{3} x_{15} +
    \\theta_{4} x_{16}


For a specially designed kernel, important information can be extracted from the
convolution of the image and the kernel. For example, convolution can be used
for edge detection in images. This means that the value of the convolution would
be large when a group of pixels contains an edge and small otherwise. The main
concept behind using convolutions in deep learning is to have these kernels be
*learnable*. Rather than having each pixel in an image assigned a learnable
parameter, the learnable parameters are the values of the convoltional kernel.
The intuition is then that a kernel can learn to extract the information from
the image that is important for the function you want to approximate. We then
define multiple kernels per layer of a network to extract a multitude of
important information from a single image.

.. admonition:: Terminology

    **Feature map**: the result of the convolution of an image with a kernel
"""

from cmacpy.nn.neural_network_layers import ConvLayer

# %%
# All of the built-in layers were created to have a similar structure and way of
# being used. As such, ``ConvLayer``, which differs from ``FCLayer`` by using
# the 2D convolution as the linear function in the layer, can be used with a lot
# of the same arguments given in :ref:`fclayer_exa` and :ref:`fclayer_adv_exa`.
# The main difference are the arguments to set up the convolution differ from
# those needed to set up the fully-connected operation. In particular,
# ``ConvLayer`` (and subsequently ``ConvTranspLayer`` and ``ResLayer``) makes
# use of Pytorch's ``nn.Conv2d`` which requires three parameters: the number of
# input channels, the number of output channels and the kernel size.
# 
# .. caution::
#     ``nn.Conv2d`` can be thought of somewhat as a misnomer -- while the
#     convolution being performed is in two spatial dimensions, the image and
#     the convolutional kernels need to be considered as three dimensional
#     objects to understand the input and output channels needed to define the
#     layer.

# sphinx_gallery_thumbnail_path = "./images/conv.png"