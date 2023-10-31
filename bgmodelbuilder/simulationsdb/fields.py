from mongoengine.fields import BaseField
import pint
import logging
from typing import Union, Optional
import numpy as np
import io
import json
from collections.abc import Mapping

from ..asymmetric import AsymmetricError
from ..common import units as unitreg
from .histogram import Histogram

log = logging.getLogger(__name__)

UnitType = Union[str, pint.Unit, pint.Quantity]


class NumpyEncoder(json.JSONEncoder):
    """ needed to json encode numpy arrays. stolen from
    https://stackoverflow.com/a/47626762/3657349
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def nparraystolists(doc: dict) -> dict:
    """ Convert all np arrays to lists for storage in db """
    for k, v in list(doc.items()):
        if isinstance(v, np.ndarray):
            doc[k] = v.tolist()
    return doc

def compress(doc: dict, force: bool = False) -> Union[dict, bytes]:
    """ If the document is smaller as a compressed npz blob than string,
    return the compressed version
    If Force is True, always return binary, otherwise test if
    json representation is smaller
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, **doc)
    if not force:
        if buf.tell() > len(json.dumps(doc, cls=NumpyEncoder)):
            # nake sure all numpy arrays are plain lists
            return nparraystolists(doc)
    return buf.getvalue()

def decompress(blob: bytes) -> dict:
    """ return a dict from the output of `compress` """
    buf = io.BytesIO(blob)
    value = dict(**np.load(buf))
    # remore np.array wrapper from non-array items
    for k, v in list(value.items()):
        try:
            value[k] = v.item()
        except (ValueError, AttributeError):
            pass
    return value

def utostr(unit):
    return '{:~C}'.format(unit)

class UnitField(BaseField):
    """ Store a unit as a string """
    def to_python(self, value):
        if isinstance(value, str):
            try:
                value = unitreg(value)
            except pint.errors.UndefinedUnitError:
                self.error(f"{value} is not a valid unit")
        if hasattr(value, 'u'):
            value = value.u
        return value

    def to_mongo(self, value):
        return utostr(value)

    def validate(self, value):
        if not isinstance(value, pint.Unit):
            self.error(f"{value} is not a pint Unit")


class QuantityField(BaseField):
    """ A field representing a pint.Quantity.
    Args:
        units: value must have same dimensionality as units
        allownone: if False, raise an error if value is None. If True,
                   allow the value to be None. If an int or float, convert
                   None to that value
        convert: if True, convert provided value to units
        forceasym: if True, force the value to be an AsymmetricError
    """
    def __init__(self, *args,
                 units: Optional[UnitType] = None,
                 allownone: Union[bool,int,float] = True,
                 convert: bool = False,
                 forceasym: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(units, 'dimensionality') and units is not None:
            units = unitreg(units)
        self.units = units
        self.allownone = allownone,
        self.convert = convert
        self.forceasym = forceasym

    def to_python(self, value):
        units = self.units

        if value is None:
            if self.allownone is True:
                return None
            elif self.allownone is False:
                raise ValueError("None is not allowed for this quantity")
            else:
                value = self.allownone

        if isinstance(value, Mapping):
            units = value.pop('units', units)
            if 'sigma' in value:
                value = AsymmetricError(**value)
            else:
                value = value['value']

        if not isinstance(value, pint.Quantity):
            value = pint.Quantity(value, units)

        if self.forceasym and not isinstance(value.m, AsymmetricError):
            value = pint.Quantity(AsymmetricError(value.m,0), value.u)

        if self.convert:
            value.ito(units)

        return value

    def to_mongo(self, value):
        result = dict(value=value.m)
        if isinstance(value.m, AsymmetricError):
            result = value.m.todict()
        if not value.dimensionless:
            result['units'] = utostr(value.u)
        #return nparraystolists(result)
        return result

    def validate(self, value):
        if value is None and self.allownone is True:
            return
        if not isinstance(value, pint.Quantity):
            self.error('Value must be a pint Quantity')
        if self.units is not None and not value.is_compatible_with(self.units):
            self.error(f'Value must have units compatible with {self.units}')
        if self.forceasym and not isinstance(value.m, AsymmetricError):
            self.error('Numeric part of quantity must be an AsymmetricError')


class UncertainQuantityField(QuantityField):
    """ A quantity field with enforced (asymmetric) uncertainty """
    def __init__(self, *args, **kwargs):
        kwargs.pop('forceasym', None)
        super().__init__(*args, **kwargs, forceasym=True)

class HistogramField(UncertainQuantityField):
    def __init__(self, *args,
                 binsunit: Optional[UnitType] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.binsunit = binsunit

    def to_python(self, value):
        if value is None:
            return value
        if isinstance(value, bytes):
            value = decompress(value)
        if isinstance(value, Mapping):
            bins = pint.Quantity(value.pop('bins'),
                                 value.pop('binsunit', self.binsunit))
            hist = QuantityField.to_python(self, value)
            value = Histogram(hist, bins)
        if not isinstance(value, Histogram):
            #shouldn't get here...
            self.error(f"Unhandled value for HistogramField {value}")
        if self.convert:
            value.bin_edges.ito(self.binsunit)
        return value

    def to_mongo(self, value):
        result = QuantityField.to_mongo(self, value.hist)
        try:
            result['bins'] = value.bin_edges.m
            if not value.bin_edges.dimensionless:
                result['binsunit'] = utostr(value.bin_edges.u)
        except AttributeError:
            result['bins'] = value.bin_edges
        return compress(result)

    def validate(self, value):
        if not isinstance(value, Histogram):
            self.error("Value must be a Histogram object")
        super(QuantityField, self).validate(value.hist)
        if len(value.hist) != len(value.bin_edges)-1:
            self.error("Histogram and bin lengths do not match")
        if self.binsunit is not None:
            try:
                if not value.bin_edges.is_compatible_with(self.binsunit):
                    self.error(f"Bins units {value.bin_edges.u} not compatible"
                               " with {self.binsunit}")
            except AttributeError:
                pass


