import numpy as np
from tqdm import tqdm
from typing import Tuple

def segmentation(img: np.ndarray, n: int) -> np.ndarray:
    '''
    This is a function that will segment the images into square segments
    with dimensions n x n.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to be segmented.
    n : int
        The dimension of the segments.
    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    '''

    N = img.shape[0] // n #the number of whole segments in the y-axis
    M = img.shape[1] // n #the number of whole segments in the x-axis

    ####
    # there are 4 cases
    #+------------+------------+------------+
    #| *n         | y segments | x segments |
    #+------------+------------+------------+
    #| N !=, M != | N+1        | M+1        |
    #+------------+------------+------------+
    #| N !=, M =  | N+1        | M          |
    #+------------+------------+------------+
    #| N =, M !=  | N          | M+1        |
    #+------------+------------+------------+
    #| N =, M =   | N          | M          |
    #+------------+------------+------------+
    ####
    if N*n != img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N+1, M+1, n, n), dtype=np.float32)
    elif N*n != img.shape[0] and M*n == img.shape[1]:
        segments = np.zeros((N+1, M, n, n), dtype=np.float32)
    elif N*n == img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N, M+1, n, n), dtype=np.float32)
    else:
        segments = np.zeros((N, M, n, n), dtype=np.float32)

    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])

    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,i*n:(i+1)*n]
            elif i == x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,-n:]
            elif i != x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,i*n:(i+1)*n]
            elif i == x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,-n:]

    segments = np.reshape(segments, newshape=((segments.shape[0]*segments.shape[1]), n, n))

    return segments

def segment_cube(img_cube: np.ndarray, n: int) -> np.ndarray:
    '''
    A function to segment a three-dimensional datacube.

    Parameters
    ----------
    img_cube : numpy.ndarray
        The image cube to be segmented.
    n : int
        The dimension of the segments.
    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image cube.
    '''

    for j, img in enumerate(tqdm(img_cube, desc="Segmenting image cube: ")):
        if j == 0:
            segments = segmentation(img, n=n)
            #we expand the segments arrays to be four-dimensional where one dimension will be the image positiion within the cube so it will be (lambda point, segments axis, y, x)
            segments = np.expand_dims(segments, axis=0)
        else:
            tmp_s = segmentation(img, n=n)
            tmp_s = np.expand_dims(tmp_s, axis=0)
            #we then add each subsequent segmented image along the wavelength axis
            segments = np.append(segments, tmp_s, axis=0)
    segments = np.swapaxes(segments, 0, 1) #this puts the segment dimension first, wavelength second to make it easier for data loaders

    return segments

def mosaic(segments: np.ndarray, img_shape: Tuple, n: int) -> np.ndarray:
    '''
    A processing function to mosaic the segments back together.

    Parameters
    ----------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    img_shape : tuple
        The shape of the original image.
    n : int
        The dimension of the segments.
    Returns
    -------
    mosaic_img : numpy.ndarray
        The reconstructed image.
    '''

    N = img_shape[0] // n
    M = img_shape[1] // n
    if N*n != img_shape[0] and M*n != img_shape[1]:
        segments = np.reshape(segments, newshape=(N+1, M+1, *segments.shape[-2:]))
    elif N*n != img_shape[0] and M*n == img_shape[1]:
        segments = np.reshape(segments, newshape=(N+1, M, *segments.shape[-2:]))
    elif N*n == img_shape[0] and M*n != img_shape[1]:
        segments = np.reshape(segments, newshape=(N, M+1, *segments.shape[-2:]))
    else:
        segments = np.reshape(segments, newshape=(N, M, segments.shape[-2:]))

    mosaic_img = np.zeros(img_shape, dtype=np.float32)
    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])
    y_overlap = img_shape[0] - N*n
    x_overlap = img_shape[1] - M*n


    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                mosaic_img[j*n:(j+1)*n,i*n:(i+1)*n] = segments[j,i]
            elif i == x_range[-1] and j != y_range[-1]:
                mosaic_img[j*n:(j+1)*n,-x_overlap:] = segments[j,i,:,-x_overlap:]
            elif i != x_range[-1] and j == y_range[-1]:
                mosaic_img[-y_overlap:,i*n:(i+1)*n] = segments[j,i,-y_overlap:]
            elif i == x_range[-1] and j == y_range[-1]:
                mosaic_img[-y_overlap:,-x_overlap:] = segments[j,i,-y_overlap:,-x_overlap:]
            else:
                raise IndexError("These indices are out of the bounds of the image. Check your ranges!")

    for j in y_range:
        for i in x_range:
            if ((j-1) >= 0) and ((i-1) >= 0) and ((j+1) <= y_range[-1]) and ((i+1) <= x_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n]
                left = mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3]
                right = mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0,top.shape[0]-1]), np.array([top[0,k],top[-1,k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0,k],bottom[-1,k]]))
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0],left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0],right[k,-1]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n] = bottom_new
                mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3] = right_new
            elif (j == 0) and ((i-1) >= 0) and ((i+1) <= x_range[-1]):
                left = mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3]
                right = mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3]

                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(left.shape[-2]):
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0], left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0], right[k,-1]]))

                mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3] = right_new

            elif (i == 0) and ((j-1) >= 0) and ((j+1) <= y_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0, top.shape[0]-1]), np.array([top[0, k], top[-1, k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0, k], bottom[-1, k]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n] = bottom_new

            elif (i == x_range[-1]) and ((j-1) >= 0) and ((j+1) <= y_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, -x_overlap:]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, -x_overlap:]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0, top.shape[0]-1]), np.array([top[0, k], top[-1, k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0, k], bottom[-1, k]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, -x_overlap:] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, -x_overlap:] = bottom_new

            elif (j == y_range[-1]) and ((i-1) >= 0) and ((i+1) <= x_range[-1]):
                left = mosaic_img[-y_overlap:, (i*n)-3:(i*n)+3]
                right = mosaic_img[-y_overlap:, ((i+1)*n)-3:((i+1)*n)+3]

                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(left.shape[-2]):
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0], left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0], right[k,-1]]))

                mosaic_img[-y_overlap:, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[-y_overlap:, ((i+1)*n)-3:((i+1)*n)+3] = right_new

    return mosaic_img

def mosaic_cube(segments: np.ndarray, img_shape: Tuple, n: int) -> np.ndarray:
    '''
    A function to mosaic a segment list into an image cube.

    Parameters
    ----------
    segments : numpy.ndarray
        The segments to be mosaiced back into images.
    img_shape : tuple
        The dimensions of the images.
    n : int
        The dimensions of the segments. Default is 64 e.g. 64 x 64.
    Returns
    -------
    m_cube : numpy.ndarray
        The cube of mosaiced images.
    '''

    m_cube = np.zeros((segments.shape[1], *img_shape), dtype=np.float32)
    segments = np.swapaxes(segments, 0, 1) #swap the number of segments and wavelength channels back to make it easier to mosaic along the wavelength axis

    for j, img in enumerate(segments):
        m_cube[j] = mosaic(img, img_shape, n=n)

    return m_cube