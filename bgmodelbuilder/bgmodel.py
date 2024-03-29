"""
@file bgmodel.py
@author: bloer

Defines the BgModel class, which collects metadata about a model definition.
The main purpose of this class is to aid in importing/exporting a complex
model to a bare dict.
"""

#python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
#from builtins import super

import copy
from shortuuid import uuid

from .mappable import Mappable
from .component import buildcomponentfromdict, Assembly, BaseComponent
from .emissionspec import EmissionSpec, buildspecfromdict
from .simulationsdb.simdatamatch import buildmatchfromdict, AssemblyPath

class BgModel(Mappable):
    def __init__(self, name=None, assemblyroot=None,
                 version="0", description='',
                 derivedFrom=None, editDetails=None,
                 components=None, specs=None, simdata=None, simsdb=None,
                 sanitize=True, *args, **kwargs):
        """Store additional info about the model for bookeeping.
        Params:
            assemblyroot (Assembly): top-level Assembly defining entire model
            name (str): a short name for the model to be used as an identifier
            version (str): version number for name
            description (str): brief description of the model contents
            derivedFrom: a reference to the parent or predecessor that was the
                base for this model, assuming it is an edit of a prior version
            editDetails (dict): metadata about the most recent edit
                (e.g. username, date, comment);
            components (dict): dictionary mapping IDs to components
            specs (dict): dictionary mapping IDs to CompSpecs
            simdata(dict): dictionary mapping IDs to SimDataMatch objects
            simsdb (str): identifier for last accessed simulation database
            sanitize(bool): add cross references, etc to model on creation
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.assemblyroot = assemblyroot or Assembly(self.name or "World")
        self.version = version
        self.description = description
        self.derivedFrom = derivedFrom
        self.editDetails = editDetails or {}
        self.components = components or {}
        self.specs = specs or {}
        self.simdata = simdata or {}
        self.simsdb = simsdb
        if sanitize:
            self.sanitize()


    def get_unplaced_components(self, toponly=True):
        """Get a list of all components not placed into any assembly
        Args:
            toponly(bool): If true, return only top-level components
                with no placements. Otherwise, return all components
                that are not a descendent of assemblyroot"""
        res = [comp for comp in self.components.values()
               if not self.assemblyroot.isparentof(comp, deep=True)]
        if toponly:
            res= [comp for comp in res  if not comp.placements]

        return res

    def getsimdata(self, component=None, spec=None, rootspec=None,
                   rootcomponent=None):
        """Get all simdata, optionally filtered by component or spec
        Args:
            component (BaseComponent): Component to filter against
                                       (leaf-level only)
            spec (EmissionSpec): Spec to filter for, must match exactly
            rootspec (EmissionSpec): Root spect to filter for
            rootcomponent (Assembly): Accept all subcomponents of base
        Returns:
            matches: generator expression or list of matching SimDataMatch
                     objects
        """
        res = self.simdata.values()
        if component is not None:
            res = (m for m in res if m.component == component)
        if spec is not None:
            res = (m for m in res if m.spec == spec )
        if rootspec is not None:
            res = (m for m in res if m.spec.getrootspec() == rootspec)
        if rootcomponent is not None:
            res = (m for m in res if rootcomponent in m.assemblyPath.components)
        return list(res)

    def getcomponents(self):
        """ get components. Simple wrapper to provide same signature
        as other methods """
        return self.components.values()

    def getspecs(self, rootonly=True):
        """Get all EmissionSpecs stored in the model.
        Args:
            rootonly (bool): if True (default), only get root-level specs
        Returns:
            list or generator expression of all specs
        """
        res = self.specs.values()
        if rootonly:
            res = (s for s in res if not s.parent)
        return res

    def registerobject(self, obj, registry):
        """ Make sure that obj has an _id attribute and add it to registry
        """
        if not hasattr(obj,'_id'):
            obj._id = uuid()
        if obj._id not in registry:
            registry[obj._id] = obj
        elif registry[obj._id] != obj:
            raise KeyError(f"Object with ID {obj._id} already registered")

    def connectreferences(self, component):
        """Convert any raw IDs in a component to actual objects"""
        if hasattr(component, 'components'):
            for placement in component.components:
                if not isinstance(placement.component, BaseComponent):
                    placement.component = self.components[placement.component]
                    placement.component.placements.add(placement)

        for boundspec in component.specs:
            if not isinstance(boundspec.spec, EmissionSpec):
                boundspec.spec = self.specs[boundspec.spec]
                boundspec.spec.appliedto.add(component)

    def sanitize(self, comp=None):
        """Make sure that all objects are fully built and registered"""

        if not comp:
            comp = self.assemblyroot
            #rebuild the registries from scratch
            #self.components.clear()
            #self.specs.clear()


        # make sure that this component has an ID and is in the registry
        self.registerobject(comp, self.components);

        # make sure all of its specs are in the registery too
        # also make sure the reverse reference to owned component exists
        for spec in comp.getspecs(deep=False, children=False):
            self.registerobject(spec, self.specs)
            # this should not be necessary, but shouldn't hurt either
            spec.appliedto.add(comp)
        # also register subspecs now
        for spec in comp.getspecs(deep=True, children=False):
            self.registerobject(spec, self.specs)

        if hasattr(comp, 'components'):
            for placement in comp.components:
                self.sanitize(placement.component)

        # miscellaneous checks
        if comp is self.assemblyroot:
            # make sure unplaced components are handled too
            for comp in self.get_unplaced_components():
                self.sanitize(comp)

            # check to see if simdata refers to nonexistent entries
            for key, match in list(self.simdata.items()):
                if match.spec not in match.component.getspecs(deep=True):
                    del self.simdata[key]
                    continue
                for plc in match.assemblyPath:
                    if not plc.parent.isparentof(plc.component):
                        del self.simdata[key]
                        break

    def delcomponent(self, compid):
        """ delete component with id `compid` completely """
        component = self.components.get(compid)
        if not component:
            return

        # remove all placements
        for placement in component.placements:
            if placement.parent:
                placement.parent.delcomponent(placement)
        # remove all sim data hits
        for match in self.getsimdata(rootcomponent=component):
            del self.simdata[match.id]
        # remove from the registry
        del self.components[compid]

    def delspec(self, specid):
        """ delete emissionspec with id `specid` completely """
        spec = self.specs.get(specid)
        if not spec:
            return

        #remove all references
        for comp in list(spec.appliedto):
            comp.delspec(spec)
        for match in self.getsimdata(rootspec=spec):
            del self.simdata[match.id]
        del self.specs[spec.id]


    @staticmethod
    def pack(obj):
        res = obj.todict()
        res.pop('_id',None)
        return res

    def todict(self, sanitize=True):
        """Export all of our data to a bare dictionary that can in turn be
        exported to JSON, stored in a database, etc
        """

        #first make sure all objects are registered and built sanely
        if sanitize:
            self.sanitize()

        #now convert all objects to dicts, and all references to IDs
        result = copy.copy(self.__dict__)

        result['assemblyroot'] = self.assemblyroot._id
        result['specs'] = \
        {key: self.pack(spec) for key, spec in self.specs.items()
         if spec.parent is None } #only do top-level specs
        result['components'] = \
        {key: self.pack(comp) for key, comp in self.components.items()}
        result['simdata'] = \
        {key: self.pack(match) for key, match in self.simdata.items()}
        # first entry in assembly paths is always assemblyroot, so can remove
        for match in result['simdata'].values():
            try:
                match['assemblyPath'].pop(0)
            except IndexError:
                pass
        return result

    @classmethod
    def buildfromdict(cls, d):
        """ Construct a new BgModel from a dictionary. It's assumed that
        the dict d was generated from the todict method previously
        """
        d['sanitize'] = False
        model = cls(**d)

        # now we need to construct specs and components from their objects
        # this does not convert ID references to objects!
        for key, spec in list(model.specs.items()):
            spec['_id'] = key
            spec = buildspecfromdict(spec)
            model.specs[key] = spec
            if hasattr(spec,'subspecs'):
                for subspec in spec.subspecs:
                    model.specs[subspec.id] = subspec
        for key, comp in model.components.items():
            comp['_id'] = key
            model.components[key] = buildcomponentfromdict(comp)
        for key, match in model.simdata.items():
            match['_id'] = key
            model.simdata[key] = buildmatchfromdict(match)

        # update the assemblyroot to the actual object
        if not isinstance(model.assemblyroot, Assembly):
            model.assemblyroot = model.components[model.assemblyroot]

        # now go through and connect all references
        # connect component references
        for comp in model.components.values():
            model.connectreferences(comp)


        # connect simdata references
        for match in model.simdata.values():
            if match.spec:
                match.spec = model.specs[match.spec]
            if match.assemblyPath:
                try:
                    root = match.assemblyPath[0]
                    match.assemblyPath[0] = model.components[root]
                except KeyError:
                    # this is the compact version
                    match.assemblyPath.insert(0, model.assemblyroot)
                match.assemblyPath = AssemblyPath.construct(match.assemblyPath)


        # should not be necessary
        #model.sanitize()

        return model


    def validate(self):
        """Make sure the model is in a cohererent state for saving"""
        # TODO: implement this
        # first, make sure all emissionspecs have correct units
        return not self.geterrors()

    def geterrors(self):
        """Return a list of error messages from validation"""
        # TODO: implement this
        return []
