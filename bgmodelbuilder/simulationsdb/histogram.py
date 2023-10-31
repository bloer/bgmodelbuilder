import operator
import numpy as np
from ..common import units, compressdict

class Histogram(object):
    # check only bins size when combining or total equality
    __strictchecking = False

    """2-tuple mimicking np.histogram structure, with operator overloads
    so that bins are not added/scaled etc
    """
    def __init__(self, hist, bin_edges=None):
        self.hist = hist
        self.bin_edges = bin_edges
        if bin_edges is None:
            self.bin_edges = np.arange(len(self.hist))


    def find_bin(self, x):
        """Find the index of the bin where x is.
        Args:
            x (float): value to search for. must have same units as bins
        Returns:
            bin (int): will be -1 if before first bin, Nbins if outside last bin
        """
        return self.bin_edges.searchsorted(x, 'right')-1

    def val(self, x, interp=None):
        """Get the value of the histogram at x. x must be in the same
        units as the bins.
        Args:
            x (float): Value to test
            interp (str): Currently: if truthy, linearly interpolate the value
                          between bins. For future use: accept string
                          specifying interpolation method
        Returns:
            val (float): value of bin where x is, None if x is outside bins
        """
        bin = self.find_bin(x)
        if x<0 or x >= len(self.hist):
            return None
        val = self.hist[bin]
        if interp and bin < len(self.hist)-1:
            slope = ( (self.hist[bin+1]-self.hist[bin]) /
                      (self.bin_edges[bin+1]-self.bin_edges[bin]) )
            val += slope * (x-self.bin_edges[bin])
        return val

    def _bound(self, a, b, bins=None):
        """coerce a and b to the edges of bins"""
        if bins is None:
            bins = self.bin_edges
        if a is None or a < bins[0]:
            a = bins[0]
        if b is None or b >= bins[-1]:
            b = bins[-1]
        return (a, b)

    def integrate(self, a=None, b=None, binwidth=True):
        """Integrate the histogram from a to b. If a and b do not correspond to
        exact bin edges, the value in the bin will be interpolated.

        Args:
            a (float): low range. If bin_edges has units, this must have the
                       same dimensionality. IF bin_edges is None, this will
                       be treated as a bin index. if None, use low bound
            b (float): upper edge. if None, use upper bound
            binwidth (bool): if True (default), multiply each bin by the bin
                             width. If False, just add bin values
        """
        bins = self.bin_edges
        spec = self.hist
        a, b = self._bound(a,b)
        weights = np.zeros_like(spec)
        first = bins.searchsorted(a,"right")-1
        last = bins.searchsorted(b,"left")-1
        weights[first:last] = 1
        #take fractions of the first and last bins
        if first == last:
            weights[first] = (b-a) / (bins[first+1] - bins[first])
        else:
            weights[first] = (bins[first+1]-a)/(bins[first+1]-bins[first])
            weights[last] = (b-bins[last]) / (bins[last+1]-bins[last])

        if binwidth:
            weights = weights * (bins[1:]-bins[:-1])

        return self.hist.dot(weights)


    def average(self, a=None, b=None, binwidths=True):
        """Calculate the average from a to b. See `integrate` for description
        of the arguments
        """
        a, b = self._bound(a,b)
        return self.integrate(a, b, binwidths) / (b-a)

    def __repr__(self):
        return f"Histogram<{len(self.hist)} bins>"

    def __str__(self):
        return repr(self)

    def _testbins(self, other):
        try:
            otherbins = other.bin_edges
        except AttributeError:
            return self.bin_edges
        if otherbins is None:
            return self.bin_edges
        elif self.bin_edges is None:
            return otherbins
        else:
            binsmatch = (self.bin_edges.shape == otherbins.shape)
            if self.__strictchecking:
                binsmatch = np.array_equal(self.bin_edges,otherbins)
            if not binsmatch:
                msg = ("Can't combins histograms with different binning: %s and %s"
                       %(self.bin_edges, otherbins))
                raise ValueError(msg)
        return self.bin_edges

    def _combine(self, other, op, inplace=False):
        #treat None as zero
        if other is None:
            other = 0
        #make sure bins are equal
        bins = self._testbins(other)

        try:
            otherhist = other.hist
        except AttributeError:
            #how to tell the difference between a 2-tuple and 1D array???
            otherhist = other
        if inplace:
            self.hist = op(self.hist, otherhist)
            return self
        else:
            return self.__class__(op(self.hist,otherhist), bins)


    #todo: should we provide for adding raw spectra rather than just Histograms?
    #binary copy operators
    def __add__(self, other):
        return self._combine(other, operator.add)
        try:
            return self._combine(other, np.add)
        except units.errors.DimensionalityError:
            return self._combine(other, operator.add)

    def __sub__(self, other):
        return self._combine(other, operator.subtract)

    def __mul__(self, other):
        return self._combine(other, operator.mul)

    def __floordiv__(self, other):
        return self._combine(other, operator.floor_divide)

    def __truediv__(self, other):
        return self._combine(other, operator.true_divide)

    def __mod__(self, other):
        return self._combine(other, operator.mod)

    def __pow__(self, other):
        return self._combine(other, operator.power)

    #do we need logical/bitwise operators??

    #binary in-place operators
    def __iadd__(self, other):
        return self._combine(other, operator.add, inplace=True)

    def __isub__(self, other):
        return self._combine(other, operator.subtract, inplace=True)

    def __imul__(self, other):
        return self._combine(other, operator.multiply, inplace=True)

    def __ifloordiv__(self, other):
        return self._combine(other, operator.floor_divide, inplace=True)

    def __itruediv__(self, other):
        return self._combine(other, operator.true_divide, inplace=True)

    def __imod__(self, other):
        return self._combine(other, operator.mod, inplace=True)

    def __ipow__(self, other):
        return self._combine(other, operator.power, inplace=True)

    #reverse binary operators
    #these should only ever be called if type(other) != type(self)
    def __radd__(self, other):
        return self.__class__(other + self.hist, self.bin_edges)

    def __rsub__(self, other):
        return self.__class__(other - self.hist, self.bin_edges)

    def __rmul__(self, other):
        return self.__class__(other * self.hist, self.bin_edges)

    def __rfloordiv__(self, other):
        return self.__class__(other // self.hist, self.bin_edges)

    def __rtruediv__(self, other):
        return self.__class__(other / self.hist, self.bin_edges)

    def __rmod__(self, other):
        return self.__class__(other % self.hist, self.bin_edges)

    def __rpow__(self, other):
        return self.__class__(other ** self.hist, self.bin_edges)



    #unary operators
    def __neg__(self):
        return self.__class__(-self.hist, self.bin_edges)

    def __abs__(self):
        return self.__class__(operator.abs(self.hist), self.bin_edges)




