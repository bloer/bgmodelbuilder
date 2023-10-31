from mongoengine import Document, DynamicDocument, EmbeddedDocument
from mongoengine.fields import (StringField, MapField, UUIDField, ListField,
                                SortedListField, IntField, DynamicField,
                                EnumField, EmbeddedDocumentField, BooleanField,
                                DateTimeField, FloatField)
from mongoengine.errors import ValidationError
from mongoengine.context_managers import switch_collection
from .fields import (QuantityField, UncertainQuantityField, HistogramField,
                     UnitField, unitreg)
import bson
from typing import Union, Optional
from enum import Enum
import logging
import datetime
log = logging.getLogger(__name__)


class NormMultiplier(Enum):
    rate = 'rate'
    flux = 'flux'
    flux_per_sr = 'flux_per_sr'
    none = 'none'

    @property
    def units(self):
        unitstr = None
        if self is NormMultiplier.rate:
            unitstr = '1/s'
        elif self is NormMultiplier.flux:
            unitstr = '1/s/cm**2'
        elif self is NormMultiplier.flux_per_sr:
            unitstr = '1/s/cm**2/sr'
        return unitreg(unitstr)

class HitEfficiency(DynamicDocument):
    """ Document describing how efficiently radiation from a given source at
    a given location is converted to hits in the sensitive detector. Usually
    these are produced by Monte Carlo simulations. This is a DynamicDocument
    so any metadata is permitted

    Required attributes:
       source (str): What is the radiation source? Match name from assay, etc.
       location (str): Where the source is located relative to the detector
                       usually the name of a MC volume
       distribution (str): whether the source is distributed in the bulk,
                           on a surface, or some other way
       norm (NormMultiplier): To convert HitEfficiency to rates in detector,
                              multiply by emission rate in some units,
                              usually 'rate' (decays/s), but can be
                              'flux' (primaries/s/cm**2)
                              'flux_per_sr' (primaries/s/cm**2/sr) or
                              'none' (HitEff is absolutely normalized)
       values (dict): Single-value hit efficiencies (as asymmetric uncertainties
                      with units), such as integral counts over some ROI
       spectra (dict): histograms of hit efficiencies

    Suggested metadata:
       nprimaries: number of primary particles simulated
       primary_spectrum: filename or representation of the spectrum of
                         particles thrown e.g. simulating (alpha,n) neutrons
       primary_particle: name of primary particle
       primary_yield: when simulating e.g. neutrons or equilibrium gammas, the
              average neutrons or gammas emitted per parent isotope decay
       biasweight: any biasing applied to the simulation
       livetime: In rare cases the simulation or spectrum is absolutely
                 normalized, e.g. coherent neutrino backgrounds or dark current
                 In this case livetime can be recorded rather than calculated
       version: software version information
       files: filenames used for calculation
       uuids: UUIDs of files used
       date: date entry was created

    If nprimaries is provided, the simulation livetime will be displayed where
    appropriate as (nprimaries*biasweight / (emissionrate*yield))

    Queries against the database are made against (source, location, distr.).
    Multiple responses are grouped by (primary_particle, primary_spectrum).
    So e.g.
    """
    # required metadata and results
    source = StringField(required=True)
    location = StringField(required=True)
    distribution = StringField(required=False, default='bulk')
    norm = EnumField(NormMultiplier, required=False,
                     default=NormMultiplier.rate)
    values = MapField(UncertainQuantityField(allownone=True),
                      required=False, default=dict)
    spectra = MapField(HistogramField(allownone=True),
                       required=False, default=dict)

    # book-keeping
    revision = IntField(required=True, default=-1)
    created = DateTimeField(required=True, default=datetime.datetime.now)
    modified = DateTimeField(required=True, default=datetime.datetime.now)

    # optional but suggested metadata
    nprimaries = IntField(required=False)
    primary_particle = StringField(required=False)
    primary_spectrum = DynamicField(required=False)
    primary_yield = FloatField(required=False, default=1)
    biasweight = FloatField(required=False, default=1)
    livetime = QuantityField(units='s', required=False)
    version = DynamicField(required=False)
    files = SortedListField(StringField(), required=False)
    uuids = SortedListField(UUIDField(), required=False)
    date = DynamicField(required=False)

    # for internal use
    values_keys = ListField(StringField())
    spectra_keys = ListField(StringField())

    meta = {
        'indexes': ['location', 'distribution', 'source', 'version', 'date',
                    'values_keys', 'spectra_keys'],
        #'db_alias': 'hiteffdb',
    }

    def clean(self):
        self.values_keys = list(self.values)
        self.spectra_keys = list(self.spectra)
        self.revision = self.revision + 1
        self.modified = datetime.datetime.now()

    @property
    def key(self):
        getattr(self, '_id', (self.source, self.location, self.distribution))

    def get_livetime(self, emissionrate=None):
        if self.livetime is not None:
            return self.livetime
        try:
            return (self.nprimaries * self.biasweight /
                    (emissionrate * self.primary_yield))
        except (AttributeError, TypeError):
            # nprimaries not provided
            pass
        except Exception:
            log.error("Error calculating livetime for HitEfficiency %s",
                      self.key)
        return None


class HitEffConfig(EmbeddedDocument):
    """ Configure settings for displaying and querying HitEfficiencies """
    key = StringField(required=True)
    display_name = StringField(required=False, default=None)
    display_unit = UnitField(required=False, default=None)
    description = StringField(default=None)
    link_spectrum = StringField(required=False, default=None)
    hide = BooleanField(required=False, default=False)

    @property
    def title(self):
        return self.display_name or self.key


class HitEffDbConfig(Document):
    """ Configure the HitEfficiency database """
    name = StringField(required=True, default='default', unique=True)
    match_distribution = BooleanField(required=False, default=True)
    #display_values = EmbeddedDocumentListField(HitEffConfig())
    #display_spectra = EmbeddedDocumentListField(HitEffConfig())
    display_values = MapField(EmbeddedDocumentField(HitEffConfig), required=False)
    display_spectra = MapField(EmbeddedDocumentField(HitEffConfig), required=False)
    extra_columns = ListField(StringField(), required=False, default=list)

    @property
    def display_columns(self):
        return ['source', 'location', 'primary_particle', 'nprimaries',
                'livetime'] + self.extra_columns

    @property
    def collection_name(self):
        return '.'.join(['hitefficiency', self.name])

    @property
    def queryset(self):
        with switch_collection(HitEfficiency, self.collection_name):
            return HitEfficiency.objects

    def validate_entry(self, hiteff: HitEfficiency, convert: bool = True):
        """ Validate the units of a HitEfficiency document """
        hiteff.validate()
        for configlist, entrylist in ((self.display_values, hiteff.values),
                                      (self.display_spectra, hiteff.spectra)):
            for config in configlist.values():
                if config.display_unit is None:
                    continue
                result = entrylist.get(config.key, None)
                if result is None:
                    continue
                testval = result.u * hiteff.norm.units
                if not testval.is_compatible_with(config.display_unit):
                    raise ValidationError(f"Entry {config.key} in {hiteff.key}"
                                          f" has wrong units {result.u}")
                if convert:
                    entrylist[config.key] = result.to(config.display_unit)
        return hiteff

    def validate_json(self, hiteff: str, convert: bool = True):
        """ validate entry from json represencation.
        If 'convert' is True (default), convert units"""
        try:
            entry = HitEfficiency.from_json(hiteff)
        except Exception as e:
            raise ValidationError from e
        return self.validate_entry(entry, convert)

    def find_entries(self, source: str, location: str,
                     distribution: Optional[str] = None,
                     idonly: bool = False, includespectra: bool = False):
        """ Find results in the database in the right collection
        if `idonly` is True, return only the id attribute and not the doc itself
        if `includespectra` is False, spectra are omitted
        """
        query = dict(source=str(source), location=str(location))
        if distribution is not None and self.match_distribution:
            query['distribution'] = str(distribution)
        result = self.queryset(**query)
        if idonly:
            result = [r.id for r in result.only('_id')]
        elif not includespectra:
            result = result.exclude('spectra')
        return result

    def get_entry(self, entryid: Union[str, bson.ObjectId],
                  includespectra: bool = True, pymongo: bool = False):
        queryset = self.queryset
        if not includespectra:
            queryset = queryset.exclude('spectra')
        if pymongo:
            queryset = queryset.as_pymongo()
        # todo: do we need to check that entryid is ObjectId and not str?
        result = queryset.with_id(entryid)
        return result

    def save_entry(self, entry):
        with switch_collection(HitEfficiency, self.collection_name):
            entry.save()
        return entry

    def update_from_collection(self, reload: bool = False,
                               dosave: bool = True) -> 'HitEffDbConfig':
        """ Inspect the entries in a collection and populate a config entry.
        This is useful to update the config view based on what's actually
        in the collection
        """
        if reload:
            try:
                self.reload()
            except Exception:
                pass
        for hiteff in self.queryset:
            for key in hiteff.values:
                self.display_values.setdefault(key, HitEffConfig(key=key))
            for key in hiteff.spectra:
                self.display_spectra.setdefault(key, HitEffConfig(key=key))
        if dosave:
            self.save()
        return self
