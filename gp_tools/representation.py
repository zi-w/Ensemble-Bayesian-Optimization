from __future__ import print_function
from itertools import chain

try:
    import itertools.zip as zip
except ImportError:
    pass

from builtins import range

import numpy as np
from scipy.sparse import csr_matrix

"""
authors: Clement Gehring
contact: gehring@csail.mit.edu
date: May 2015
"""

################## VARIOUS INTERFACES #######################################
""" Generic map from a state (as an array), to a feature vector (as an array)
"""


class Projector(object):
    def __init__(self):
        pass

    """ Project a vector (or matrix of row vectors) to a corresponding
        feature vector. It should handle 1-D arrays and 2-D arrays.
    """

    def __call__(self, state):
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def size(self):
        raise NotImplementedError("Subclasses should implement this!")


""" Generic map from a state-action pair (as two arrays), to a feature
    vector (as an array)
"""


class StateActionProjector(object):
    def __init__(self):
        pass

    """ Project two vectors (or two matrices of row vectors) to a corresponding
        feature vector. It should handle 1-D arrays and 2-D arrays for both
        arguments and any combination of these cases, i.e., one state but
        several actions and vice-versa.
    """

    def __call__(self, state, action):
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def size(self):
        raise NotImplementedError("Subclasses should implement this!")


""" Hashing an array of indices to a single index.
    Mostly used for tile coding.
"""


class Hashing(object):
    def __init__(self, **kargs):
        pass

    """ Hash several indices (typically, one per dimension) onto
        one index (typically, index of a tile). This could be a simple
        cartesian-product, i.e., unique index for every combination, or
        some sort of randomized hash function, e.g., UNH.

        Must be able to deal with 2D arrays of indices.
    """

    def __call__(self, indices):
        raise NotImplementedError("Subclasses should implement this!")


################## HELPER CLASSES FOR STATE PROJECTOR TO #####################
################## STATE-ACTION PROJECTOR CONVERSION     #####################

""" Simple method that ensures both arrays are 2D and that they share the same
    number of rows (for when only one row was given for one argument but not
    the other).
"""


def tile_and_adjust_state_action(state, action):
    # make everything 2-D
    if state.ndim == 1:
        state = state.reshape((1, -1))
    if action.ndim == 1:
        action = action.reshape((1, -1))

    # if one is 2D and but not the other, tile such that the dimensions
    # match.
    if state.shape[0] == 1 and action.shape[0] > 1:
        state = np.tile(state, (action.shape[0], 1))
    elif action.shape[0] == 1 and state.shape[0] > 1:
        action = np.tile(action, (state.shape[0], 1))

    return state, action


""" Simple state-action projector that simply treats the actions as
    extra dimensions and feeds it to a projector.
"""


class ConcatStateAction(StateActionProjector):
    def __init__(self, projector):
        self.projector = projector

    def __call__(self, state, action):
        state, action = tile_and_adjust_state_action(state, action)

        sa = np.hstack((state, action))
        return self.projector(sa)

    @property
    def size(self):
        return self.projector.size


""" Simple state-action projector that simply ignores the action. Mainly
    to be used when only a value function is needed.
"""


class RemoveAction(StateActionProjector):
    def __init__(self, projector):
        self.projector = projector

    def __call__(self, state, action):
        state, action = tile_and_adjust_state_action(state, action)
        return self.projector(state)

    @property
    def size(self):
        return self.projector.size


""" Create a tabular actions representaton with a state projector. The output
    vectors are padded with zeros such that the total dimension is
    n*num_actions, where n is the output dimension of the projector.

    If action i is given, then the whole vector is zero with the exception
    of columns n*i to n*(i+1), where the projected state will be encoded.

    The resulting output can be either dense or sparse.

"""


class TabularAction(StateActionProjector):
    def __init__(self, projector, num_actions, sparse=True):
        self.phi = projector
        self.__size = self.phi.size * num_actions
        self.num_actions = num_actions
        self.sparse = sparse

    def __call__(self, state, action):
        state, action = tile_and_adjust_state_action(state, action)
        phi_s = self.phi(state)
        phi_s = csr_matrix(phi_s)

        # this step assumes that, if sparse, each row has the same number of
        # non zero elements.
        action = np.tile(action, (1, phi_s.indptr[1] - phi_s.indptr[0])).reshape(-1).astype('int')

        phi_sa = csr_matrix((phi_s.data,
                             phi_s.indices + action * self.phi.size,
                             phi_s.indptr),
                            shape=(phi_s.shape[0], self.size))
        if not self.sparse:
            phi_sa = phi_sa.toarray()
        return phi_sa

    @property
    def size(self):
        return self.__size


################## HELPER CLASS FOR INDEX TO VECTOR CONVERSION ################

""" Projector converting a projector, which generate indices, to a
    projector generating sparse vectors (as a csr matrix)
"""


class IndexToBinarySparse(Projector):
    """ Constructor to go from a projector generating indices to a
        projector generating sparse vectors (as a csr matrix)

        index_projector:    the projector generating indices, it needs to be
                            be able to handle 2-D arrays
    """

    def __init__(self, index_projector, normalize=False):
        super(IndexToBinarySparse, self).__init__()

        self.index_projector = index_projector
        self.normalize = normalize
        if normalize:
            self.entry_value = 1.0 / np.sqrt(self.index_projector.nonzeros)
        else:
            self.entry_value = 1.0

    def __call__(self, state):
        # generate indices for a single (or several) binary sparse vector(s).
        indices = self.index_projector(state)
        if indices.ndim == 1:
            indices = indices.reshape((1, -1))

        # set value of all active features
        vals = np.empty(indices.size)
        vals[:] = self.entry_value

        # create row pointers, this assumes each row has the same number
        # of non-zero entries
        row_ptr = np.arange(0, indices.size + 1, indices.shape[1])

        # flatten the column indices generate ealier
        col_ind = indices.flatten()

        return csr_matrix((vals, col_ind, row_ptr),
                          shape=(indices.shape[0], self.index_projector.size))

    @property
    def size(self):
        return self.index_projector.size

    @property
    def xTxNorm(self):
        return 1.0 if self.normalize else self.index_projector.nonzeros


""" Projector converting a projector, which generate indices, to a
    projector generating dense vectors (as an array)
"""


class IndexToDense(Projector):
    """ Constructor to go from a projector generating indices to a
        projector generating dense vectors (as an array)

        index_projector:    the projector generating indices, it needs to be
                            be able to handle 2-D arrays
    """

    def __init__(self, index_projector, normalize=False):
        super(IndexToDense, self).__init__()

        self.index_projector = index_projector
        self.normalize = normalize
        if normalize:
            self.entry_value = 1.0 / np.sqrt(self.index_projector.nonzeros)
        else:
            self.entry_value = 1.0

    def __call__(self, state):
        # generate indices for a single (or several) binary vectors
        indices = self.index_projector(state)
        if indices.ndim == 1:
            indices = indices.reshape((1, -1))

        # allocate dense array
        output = np.zeros((indices.shape[0], self.size))

        # create row indices
        row_ind = np.tile(np.arange(indices.shape[0]).reshape((-1, 1)),
                          (1, indices.shape[1])).flatten()

        # set values for all active features
        output[row_ind, indices.flatten()] = self.entry_value

        # squeeze out useless dimensions, if any
        return output.squeeze()

    @property
    def size(self):
        return self.index_projector.size

    @property
    def xTxNorm(self):
        return 1.0 if self.normalize else self.index_projector.nonzeros


""" Projector concatenating several projectors into the same representation
    by concatenating their outputs.
"""


class ConcatProjector(Projector):
    def __init__(self, projectors):
        super(ConcatProjector, self).__init__()
        self.phis = projectors

    def __call__(self, state):
        return np.hstack((phi(state) for phi in self.phis))

    @property
    def size(self):
        return sum([phi.size for phi in self.phis])


class SparseRPTilecoding(IndexToBinarySparse):
    def __init__(self, index_projector, random_proj, normalize=False, output_dense=True):
        super(SparseRPTilecoding, self).__init__(index_projector, normalize=normalize)
        self.random_proj = random_proj
        self.output_dense = output_dense

    def __call__(self, X):
        phi = super(self.__class__, self).__call__(X)
        pphi = self.random_proj.dot(phi.T).T
        if self.output_dense:
            pphi = pphi.todense()
        return pphi

    @property
    def size(self):
        return self.random_proj.shape[0]


################## TILE CODING KERNEL FUNCTION ###############################
class DenseKernel(IndexToBinarySparse):
    def __call__(self, X1, X2=None):
        phi1 = super(self.__class__, self).__call__(X1)
        if X2 is None:
            return phi1.dot(phi1.T).todense()
        else:
            phi2 = super(self.__class__, self).__call__(X2)
            return phi1.dot(phi2.T).todense()


class DenseL1Kernel(object):
    def __init__(self, z, k, scale=1.0):
        self.z = z.copy()
        self.all_cat = np.unique(self.z)
        self.n_cat = self.all_cat.shape[0]
        self.k = k * 1.0
        self.scale = scale * 1.0

    def __call__(self, X1, X2=None):
        if X2 is None:
            if X1.ndim == 1:
                return self.scale
            K = 0
            for a in self.all_cat:
                active = self.z == a
                K += computeKmm(X1[:, active], self.k[active], self.scale / self.n_cat)
            return K
        else:
            X1 = X1[None, :] if X1.ndim == 1 else X1
            X2 = X2[None, :] if X2.ndim == 1 else X2
            K = 0
            for a in self.all_cat:
                active = self.z == a
                K += computeKnm(X1[:, active], X2[:, active], self.k[active], self.scale / self.n_cat)
            return K

    @property
    def xTxNorm(self):
        return self.scale


# L1 kernel helpers:
def computeKmm(Xbar, k, scale):
    Xbar = Xbar * k
    Qbar = (Xbar ** 2).sum(axis=1)[:, None]
    l2dist = Qbar + Qbar.T - 2 * (Xbar.dot(Xbar.T))
    l2dist[l2dist < 0] = 0
    ret = scale * np.exp(-np.sqrt(l2dist))
    return ret


def computeKnm(X, Xbar, k, scale):
    X = X * k
    Xbar = Xbar * k
    Q = (X ** 2).sum(axis=1)[:, None]
    Qbar = (Xbar ** 2).sum(axis=1)[:, None]
    l2dist = Q + Qbar.T - 2 * (X.dot(Xbar.T))
    l2dist[l2dist < 0] = 0

    ret = scale * np.exp(-np.sqrt(l2dist))
    return ret


class SparseKernel(IndexToBinarySparse):
    def __call__(self, X1, X2=None):
        phi1 = super(self.__class__, self).__call__(X1)
        if X2 is None:
            return phi1.dot(phi1.T)
        else:
            phi2 = super(self.__class__, self).__call__(X2)
            return phi1.dot(phi2.T)


################## TILE CODING IMPLEMENTATION ################################

""" Represents a series of layer of tile coding. This is equivalent to a single
    discretization of the input space.
"""


class Tiling(object):
    """ Constructor for a set of tilings.

        input_index: array (or list) of the input indices to be considered. This allows
                     to specify on which inputs are the tilings defined.

        ntiles: integer, or array of integers, specifying how many uniform
                divisions for each dimension (or all dimensions if a single
                integer is given). Each layer in this set will have the same
                number of divisions in each dimension.

        ntilings: The number of individual layers in this set of tilings.

        state_range:    range of each dimension

        offset: (optional) the offsets between each layer in this set of
                tilings. By default, each layer is uniformly offset from
                each other. Shape: (#dimensions, ntilings), i.e. for each
                dimension you have to specify the offset of each tiling. For
                dimension d, the offset should be negative and > -1/ntiles[d].
                So if you want random offsets for one dimension, you could use
                something like this:
                -1.0/ntiles[d] * np.random.random_sample(size=ntilings)

        hashing:    (optional) map from the individual bin index for each
                    dimension to a tile index. By default, this is assumed
                    to be a cartesian product, i.e., each combination if
                    mapped to a unique index. This is equivalent to laying a
                    grid over all input dimensions at once. Alternatively,
                    this could map could be defined by a random hash funciton
                    (e.g., UNH).
    """

    def __init__(self,
                 input_index,
                 ntiles,
                 ntilings,
                 state_range,
                 rnd_stream,
                 offset=None,
                 hashing=None):

        self.hashing = hashing

        if isinstance(ntiles, int):
            ntiles = np.array([ntiles] * len(input_index), dtype='int')
        else:
            ntiles = np.array(ntiles)

        self.state_range = [state_range[0][input_index].copy().astype(float)[None, :, None],
                            state_range[1][input_index].copy().astype(float)[None, :, None]]

        if ntiles.ndim > 1:
            ntiles = ntiles[None, :, :]
        else:
            ntiles = ntiles[None, :, None]

        self.state_range[0] = self.state_range[0] - (self.state_range[1] - self.state_range[0]) / (ntiles - 1)

        self.offset = offset
        if offset is None:
            self.offset = np.empty((ntiles.shape[1], ntilings))
            for i in range(ntiles.shape[0]):
                self.offset[i, :] = -rnd_stream.random_sample(ntilings) / ntiles[0, i]

        if self.hashing == None:
            self.hashing = IdentityHash(ntiles)

        self.input_index = np.array(input_index, dtype='int')
        self.size = ntilings * (self.hashing.memory)
        self.index_offset = (self.hashing.memory * np.arange(ntilings)).astype('int')
        self.ntiles = ntiles

    def __call__(self, state):
        return self.getIndices(state)

    def getIndices(self, state):
        if state.ndim == 1:
            state = state.reshape((1, -1))[:, :, None]
        else:
            state = state[:, :, None]

        nstate = (state[:, self.input_index, :] - self.state_range[0]) / (self.state_range[1] - self.state_range[0])
        indicies = ((self.offset[None, :, :] + nstate) * self.ntiles).astype(np.int)
        return self.hashing(indicies) + self.index_offset[None, :]

    @property
    def ntilings(self):
        return self.offset.shape[1]


""" Full tile coding implementation. This represents a projector, from states
    to features.
"""


class TileCoding(Projector):
    """ Constructor for a tile coding projector. The constructor builds
        several sets of individual tilings based on the input arguments.

        input_indicies: a list of arrays of indices. Each array of indicies
                        specifies which input dimensions are considered by
                        each set of tilings. There will be as many sets of
                        tilings as there are array of indices.

                        e.g., input_indices = [ [0,1], [1,2] ] will generate
                        two sets of tilings, one defined on the first and
                        second dimension, and the other on the second and
                        third dimension.

        ntiles: list of a mix of integers or array of integers. This specifies
                the how fine is the discretization in each set of tilings. There
                should be an element (either integer or array of integers) for
                each set of tilings. If a set of tilings is given an integer,
                each dimensions are discretized in that many bins. If a set of
                tilings is given an array, it should be of the same size as
                the number of input dimensions it uses. In this case, it will
                discretize each dimensions in as many bins as the corresponding
                integer in the given array.

                e.g., ntiles = [ 4, [2,6] ] will generate two sets of tilings
                where the first discretizes all its input dimensions in 4 bins,
                and the second discretizes its first input dimension in 2 bins
                and its second, in 6 bins.

        ntilings:   array (or list) of integers corresponding to how many
                    individual layers are in each set of tilings. In this
                    implementation, individual layers in the same set are
                    uniformly offset from each other.

        hashing:    either None, or list of hashing functions. This specifies
                    the hashing function to be used by each set of tilings. It
                    is assumed that each individual layer part of the same set
                    share the same hash funciton. If None is given, then each
                    set of tilings will use a cartesian product, i.e., each
                    combination of indices is mapped to a unique tile. This is
                    equivalent to laying a n-d grid on the input dimensions.

        state_range:    range of each dimension

        offsets:    (optional) the offsets between the layers for each set of
                    tilings. By default, all layers are uniformly offset from
                    each other. If you provide a list of lists of offsets (which
                    is recommended), this must hold: len(offsets) ==
                    len(input_indices). Each item in offsets is passed to the
                    constructor of Tiling, so see there for further
                    documentation.

        bias_term:  (optional) boolean specifying whether to add an extra bias term which
                    is always on. By default, a bias_term is added.


    """

    def __init__(self,
                 input_indices,
                 ntiles,
                 ntilings,
                 hashing,
                 state_range,
                 rnd_stream=None,
                 offsets=None,
                 bias_term=True):
        super(TileCoding, self).__init__()
        if hashing == None:
            hashing = [None] * len(ntilings)
        if offsets is None:
            offsets = [None] * len(input_indices)

        if offsets is None and rnd_stream is None:
            raise Exception(
                'Either offsets for each tiling or a random stream (numpy) needs to be given in the constructor')

        self.state_range = np.array(state_range)
        self.tilings = [Tiling(in_index, nt, t, self.state_range, rnd_stream, offset=o, hashing=h)
                        for in_index, nt, t, h, o
                        in zip(input_indices, ntiles, ntilings, hashing, offsets)]
        self.__size = sum(map(lambda x: x.size, self.tilings))
        self.bias_term = bias_term
        self.index_offset = np.zeros(len(ntilings), dtype='int')
        self.index_offset[1:] = np.cumsum(map(lambda x: x.size, self.tilings[:-1]))
        self.index_offset = np.hstack([np.array([off] * t, dtype='int')
                                       for off, t in zip(self.index_offset, ntilings)])

        if bias_term:
            self.index_offset = np.hstack((self.index_offset, np.array(self.__size, dtype='int')))
            self.__size += 1

        self.__size = int(self.__size)

    """ Map a state vector, or several state vectors, to its corresponding
        tile indices.
    """

    def __call__(self, state):
        if state.ndim == 1:
            state = state.reshape((1, -1))

        # add bias term if needed, concatenate set of indices of all
        # the sets of tilings.
        if self.bias_term:
            indices = np.hstack(chain((t(state) for t in self.tilings),
                                      [np.zeros((state.shape[0], 1), dtype='int')])) \
                      + self.index_offset
        else:
            indices = np.hstack((t(state) for t in self.tilings)) \
                      + self.index_offset

        return indices.squeeze()

    @property
    def size(self):
        return self.__size

    @property
    def nonzeros(self):
        return np.sum([t.ntilings for t in self.tilings]) + (1 if self.bias_term else 0)


class UNH(Hashing):
    # constants were taken from rlpark's implementation.
    increment = 470

    def __init__(self, memory, rnd_stream):
        super(UNH, self).__init__()
        self.rndseq = np.zeros(16384, dtype='int')
        self.memory = int(memory)
        for i in range(4):
            self.rndseq = self.rndseq << 8 | rnd_stream.random_integers(np.iinfo('int16').min,
                                                                        np.iinfo('int16').max,
                                                                        16384) & 0xff

    def __call__(self, indices):
        rnd_seq = self.rndseq
        a = self.increment * np.arange(indices.shape[1])
        index = indices + a[None, :, None]
        index = index - (index.astype(np.int) / rnd_seq.size) * rnd_seq.size
        hashed_index = (np.sum(rnd_seq[index], axis=1)).astype(np.int)
        return (hashed_index - (hashed_index / self.memory).astype(np.int) * self.memory).astype('int')


class IdentityHash(Hashing):
    def __init__(self, dims, wrap=False):
        super(IdentityHash, self).__init__()
        self.memory = np.prod(dims)
        self.dims = dims.astype('int')
        self.wrap = wrap
        self.dim_offset = np.cumprod(np.vstack((np.ones((self.dims.shape[2], 1)), self.dims[0, :0:-1, :])),
                                     axis=0).astype('int')[None, ::-1, :]

    def __call__(self, indices):
        if self.wrap:
            indices = np.remainder(indices, self.dims)
        else:
            indices = np.clip(indices, 0, self.dims - 1)
        return np.sum(indices * self.dim_offset, axis=1)


################## END OF TILE CODING IMPLEMENTATION #########################


################## RBF IMPLEMENTATION ########################################

class RBFCoding(Projector):
    """ Constructor for an RBF encoding.

        stddev: scaling of the dimensions when computing the distance. Each
                dimension needs a scale. If only a 1-D array is given, all
                RBFs are assumed to have the same scaling, otherwise, it is
                assumed that there is a row specifying the scale for each
                RBF.

        c:  centers of the RBFs. The number of rows corresponds to the number
            of RBFs. The number of column should be equal to the input
            dimension. Each row is a center for a RBF.

        normalized: Boolean to decided whether the RBFs should be normalized.

        bias_term:  Boolean to decided whether the output should be augmented
                    with a constant 1.0 bias term.
    """

    def __init__(self,
                 widths,
                 centers,
                 normalized=False,
                 bias_term=True,
                 **params):
        super(RBFCoding, self).__init__()

        # the centers of the rbfs
        self.c = centers.T[None, :, :]

        # the widths of the rbfs, each rbf can have different widths
        if widths.ndim == 1:
            self.w = widths[None, :, None]
        else:
            self.w = widths.T[None, :, :]

        # should the output of the rbfs sum to one
        self.normalized = normalized

        # include a bias term (always equal to one)
        self.bias_term = bias_term

        # size of the encoded vectors
        self.__size = centers.shape[0]

        # if bias term is included, increment the size
        if bias_term:
            self.__size += 1

    def __call__(self, state):
        # if only on 1-D array given, reshape to a compatible shape
        if state.ndim == 1:
            state = state.reshape((1, -1))

        # allocate and set bias term if needed
        last_index = self.size
        output = np.empty((state.shape[0], self.size))
        if self.bias_term:
            last_index -= 1
            output[:, -1] = 1.0

        # compute squared weighted distance distance
        dsqr = -(((state[:, :, None] - self.c) / self.w) ** 2).sum(axis=1)
        if self.normalized:
            # compute the normalized rbfs from the distances
            e_x = np.exp(dsqr - dsqr.min(axis=1)[:, None])
            output[:, :last_index] = e_x / e_x.sum(axis=1)[:, None]
        else:
            # compute the rbfs from the distances
            output[:, :last_index] = np.exp(dsqr)

        # return encoded input, squeeze out extra dimensions (in the case
        # only on input row was given)
        return output.squeeze()

    @property
    def size(self):
        return self.__size


""" Method to generate grids of points (typically for RBF coding).

    state_range: range of each dimension

    num_centers:    An integer or an array (or list) of integers which
                    corresponds the number of points to distribute on each
                    dimensions. If a single integer is given, all dimensions
                    will have the same number of points.
"""


def grid_of_points(state_range, num_centers):
    if isinstance(num_centers, int):
        num_centers = [num_centers] * state_range[0].shape[0]
    points = [np.linspace(start, stop, num, endpoint=True)
              for start, stop, num in zip(state_range[0],
                                           state_range[1],
                                           num_centers)]

    points = np.meshgrid(*points)
    points = np.concatenate([p.reshape((-1, 1)) for p in points], axis=1)
    return points

################## END OF RBF IMPLEMENTATION ##################################
