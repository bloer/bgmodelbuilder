"""
Common functions and utility classes shared by other units
"""
import inspect
from copy import copy
import pint
from pint.numpy_func import implements, unwrap_and_wrap_consistent_units
import uncertainties
import numpy as np

# physical units #######
units = pint.UnitRegistry()
units.auto_reduce_dimensions = False  # this doesn't work right
units.errors = pint.errors
units.default_format = '~gP'
# fix Bq, add ppb units
units.load_definitions([
    "Bq = Hz = Bq = Becquerel",
    "ppb_U = 12 * mBq/kg = ppbU",
    "ppb_Th = 4.1 * mBq/kg = ppbTh",
    "ppm_K = 31 * mBq/kg = ppbK",
    "ppb_K = 0.001 * ppm_K = ppbK",
    "ppt_U = 0.001 * ppb_U = pptU",
    "ppt_Th = 0.001 * ppb_Th = pptTh",
    "ppt_K = 0.001 * ppb_K = pptK",
    "dru = 1./(kg * keV * day) = dru = DRU",
    "kky = kg * keV * year = kky = kg_keV_yr",
    "ppm = 1e-6 = ppm = parts_per_million",
    "ppb = 1e-9 = ppb = parts_per_billion",
    "ppt = 1e-12 = ppt = parts_per_trillion",
])


# monkey-punch round() capability onto uncertainties, mostly needed for tests
def __rnd__(self, n=0):
    return round(self.n, n)


uncertainties.core.Variable.__round__ = __rnd__
uncertainties.core.AffineScalarFunc.__round__ = __rnd__

# monkey-patch bug when comparing quantities to measurements

def _compare(self, other, op):
    if not isinstance(other, pint.Quantity):
        if self.dimensionless:
            return op(
                self._convert_magnitude_not_inplace(self.UnitsContainer()), other
            )
        elif pint.quantity.zero_or_nan(other, True):
            # Handle the special case in which we compare to zero or NaN
            # (or an array of zeros or NaNs)
            if self._is_multiplicative:
                # compare magnitude
                return op(self._magnitude, other)
            else:
                # compare the magnitude after converting the
                # non-multiplicative quantity to base units
                if self._REGISTRY.autoconvert_offset_to_baseunit:
                    return op(self.to_base_units()._magnitude, other)
                else:
                    raise pint.OffsetUnitCalculusError(self._units)
        else:
            raise ValueError("Cannot compare Quantity and {}".format(type(other)))

    if self._units == other._units:
        return op(self._magnitude, other._magnitude)
    if self.dimensionality != other.dimensionality:
        raise pint.DimensionalityError(
            self._units, other._units, self.dimensionality, other.dimensionality
        )
    return op(self.to_root_units().magnitude, other.to_root_units().magnitude)

pint.Quantity.compare = _compare

# implement array_equal for quantities
@implements("array_equal", "function")
def _array_equal(*args, **kwargs):
    arrays, output_wrap = unwrap_and_wrap_consistent_units(*args)
    return np.array_equal(*arrays)

def ensure_quantity(value, defunit=None, convert=False):
    """Make sure a variable is a pint.Quantity, and transform if unitless

    Args:
        value: The test value
        defunit (str,Unit, Quanty): default unit to interpret as
        convert (bool): if True, convert the value to the specified unit
    Returns:
        Quantity: Value if already Quantity, else Quantity(value, defunit)
    """
    if value is None:
        return None
    try:
        qval = units.Quantity(value)
    except Exception:
        # Quantity can't handle '+/-' that comes with uncertainties...
        valunit = value.rsplit(' ', 1)
        q = valunit[0]
        u = valunit[1] if len(valunit) > 1 else ''
        if q.endswith(')'):
            q = q[1:-1]
        qval = units.Measurement(uncertainties.ufloat_fromstr(q), u)

    # make sure the quantity has the same units as default value
    if (defunit is not None and
        qval.dimensionality != units.Quantity(1*defunit).dimensionality):
        if qval.dimensionality == units.dimensionless.dimensionality:
            qval = units.Quantity(qval.m, defunit)
        else:
            raise units.errors.DimensionalityError(qval.u, defunit)

    return qval.to(defunit) if convert and defunit else qval

class _Stringify(object):
    """ Convert objects to strings so that they can be reconstructed later
    Currently only applies to pint Quantities
    """

    @staticmethod
    def appliesto(val):
        return isinstance(val, pint.Quantity)

    @staticmethod
    def stringify(val):
        return _Stringify.stringify_quantity(val)

    @staticmethod
    def stringify_quantity(val):
        return "{} {:~P}".format(repr(val.m), val.u)


def to_primitive(val, renameunderscores=True, recursive=True,
                 replaceids=True):
    """Transform a class object into a primitive object for serialization"""

    #if replaceids and hasattr(val, 'id'):
    #    val =  val.id   #id can be a property, so only use _id
    if replaceids and hasattr(val, '_id'):
        val =  val._id

    elif inspect.getmodule(val): #I think this tests for non-builtin classes
        if hasattr(val, 'todict'): #has a custom conversion
            val =  val.todict()
        elif _Stringify.appliesto(val):
            val =  _Stringify.stringify(val)
        elif hasattr(val, '__dict__'): #this is probably going to break lots
            val =  copy(val.__dict__)
        else: #not sure what this is...
            raise TypeError("Can't convert %s to exportable",type(val))

    if recursive:
        if isinstance(val, dict):
            removeclasses(val, renameunderscores, recursive, replaceids)
        elif isinstance(val, (list,tuple)):
            val = type(val)(to_primitive(sub, renameunderscores, recursive,
                                         replaceids) for sub in val)
    return val


####### Functions for dictionary export of complex structures #####
def removeclasses(adict, renameunderscores=True, recursive=True,
                  replaceids=True):
    """Transform all custom class objects in the dict to plain dicts
    Args:
        adict (dict): The dictionary to update in place
        renameunderscores (bool): For any key starting with a single underscore,
            replace with the regular. I.e. '_key'->'key'. Note '_id' will
            NOT be replaced
        recursive (bool): call removeclasses on any dicts stored as objects
            inside this dictionary
        replaceids (bool): If true, replace any object with an 'id' attribute
            by that object's id
    """

    underkeys = []

    for key, val in adict.items():
        #check if we need to rename this key
        if (key != '_id' and hasattr(key,'startswith') and
            key.startswith('_') and not key.startswith('__')):
            underkeys.append(key)

        adict[key] = to_primitive(val, renameunderscores, recursive,
                                  replaceids)

    if renameunderscores:
        for key in underkeys:
            adict[key[1:]] = adict[key]
            del adict[key]

    return adict


def stripdefaults(target, removeempty=list(), removeexact=dict()):
    """ Make output dicts smaller by removing default entries. Keys are
    removed from `target` in place
    Args:
        target (dict): dictionary to remove entries from
        removeempty (list): Remove keys in this list from target if the key
                            tests as false (None, "", [], {}, etc)
        removeexact (dict): Remove keys in this dict from target if the
                            correspondong values are equal
    Returns:
        target with the keys removed
    """
    for key in removeempty:
        # if val is an np.array, `not` raises an exception
        try:
            if not target.get(key, True):
                del target[key]
        except Exception:
            pass

    for key, val in removeexact.items():
        if target.get(key) == val:
            del target[key]

    return target

def try_reduce(reducer, val1, val2, fallbackval1=True):
    if val2 is None:
        return val1
    elif val1 is None:
        return val2

    try:
        return reducer(val1, val2)
    except Exception as e:
        log.warn(f"Caught exception reducing {val1} and {val2}: {e}")
        return val1 if fallbackval1 else val2
