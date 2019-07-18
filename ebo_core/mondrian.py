import ebo_core.helper as helper
import numpy as np


class MondrianNode(object):
    def __init__(self, X, y, x_range, reference=None):
        self.X = X
        self.y = y
        self.reference = reference
        self.totlen = (x_range[1] - x_range[0]).sum()
        self.x_range = x_range
        self.get_vol()
        assert self.totlen > 0, 'Node is empty. Totoal length of the search space is 0.'
        self.datasize = y.shape[0]
        self.left = None
        self.right = None
        self.epsilon = 0.  # overlap of the leaves

        self.maxdata = 1

    def get_vol(self):
        if self.reference is not None:
            self.maxy = self.y.max() - self.reference if self.y.shape[0] > 0 else 0
            self.volumn = np.exp(np.log((self.x_range[1] - self.x_range[0])).sum()) + self.maxy
        else:
            self.volumn = np.exp(np.log((self.x_range[1] - self.x_range[0])).sum())

    def partition(self):
        prob = self.x_range[1] - self.x_range[0]
        assert prob.dtype == float, 'Forgot to set x_range to be float?'
        d = helper.sample_categorical(prob)
        cut = np.random.uniform(self.x_range[0, d], self.x_range[1, d])
        leftinds = np.where(self.X[:, d] <= cut + self.epsilon)
        rightinds = np.where(self.X[:, d] >= cut - self.epsilon)
        left_range, right_range = self.x_range.copy(), self.x_range.copy()
        left_range[1, d] = cut
        right_range[0, d] = cut
        self.left = MondrianNode(self.X[leftinds], self.y[leftinds], left_range, self.reference)
        self.right = MondrianNode(self.X[rightinds], self.y[rightinds], right_range, self.reference)
        return self.left, self.right

    def delete_data(self):
        self.X = None
        self.y = None


class MondrianTree(object):
    def __init__(self, X, y, x_range, poolsize, reference=None):
        self.X = X
        self.y = y
        self.x_range = x_range
        self.root = MondrianNode(X, y, x_range, reference)
        self.poolsize = poolsize
        self.leaves = None

    def grow_tree(self, min_leaf_size):
        leaves = [self.root]
        flag = True
        while flag:
            if len(leaves) >= self.poolsize:
                break
            prob = np.array([[node.totlen, node.datasize] \
                             for node in leaves])
            mask = np.maximum(prob[:, 1] - min_leaf_size, 0)
            if mask.sum() == 0:
                print
                'Mondrian stopped at ', str(len(leaves)), ' number of leaves.'
                break

            prob = mask * prob[:, 0]
            nodeidx = helper.sample_categorical(prob)
            chosen_leaf = leaves[nodeidx]
            left, right = chosen_leaf.partition()
            leaves[nodeidx] = left
            leaves += [right]
            chosen_leaf.delete_data()

        self.leaves = leaves
        return leaves

    def update_leaf_data(self, X, y, ref=None):
        self.X, self.y = X, y
        for n in self.leaves:
            inds = np.where(np.logical_and(X <= n.x_range[1], X >= n.x_range[0]).all(axis=1))
            n.X = X[inds]
            n.y = y[inds]
            n.reference = ref
            n.get_vol()
        return self.leaves

    def visualize(self):
        if self.leaves is None or self.X.shape[1] != 2:
            print
            'error: x shape is wrong or leaves is none.'

        # visaulize 2d mondrians
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib
        font = {'size': 20}
        matplotlib.rc('font', **font)
        mondrian_colors = np.array([[255, 240, 1], [48, 48, 58],
                                    [255, 1, 1], [1, 1, 253], [249, 249, 249]])
        mondrian_colors = mondrian_colors / 255.0
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        print('number of leaves = {}'.format(len(self.leaves)))
        for node in self.leaves:
            xy = node.x_range[0]
            xylen = node.x_range[1] - node.x_range[0]
            c = mondrian_colors[4]
            p = patches.Rectangle(
                xy, xylen[0], xylen[1],
                facecolor=c,
                linewidth=1,
                edgecolor='k'
            )
            ax.add_patch(p)
        for x in self.X:
            c = '#fdbf6f'
            p = patches.Circle(
                x, 0.01,
                facecolor=c,
                linewidth=0
            )
            ax.add_patch(p)
        return ax, fig
