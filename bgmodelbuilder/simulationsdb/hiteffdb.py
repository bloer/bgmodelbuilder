from .simulationsdb import SimulationsDB
from .hiteff import NormMultiplier
import mongoengine

def addnone(a, b):
    """ return a+b unless one is none """
    if a is None:
        return b
    if b is None:
        return a
    return a + b

class HitEffDB(SimulationsDB):
    """ A stopgap implementation of SimulationsDB using HitEffConfigs.
    We trade off needing to write python code to explain our query and data
    structure by instead requiring a fixed data structure, and entries
    in the DB to define display units
    """

    def __init__(self, dbconfig, app=None, *args, **kwargs):
        self.dbconfig = dbconfig
        super().__init__(app, *args, **kwargs)

    def init_app(self, app):
        super().init_app(app)
        self.connect()

    @staticmethod
    def connect(app):
        defaulturi = "mongodb://127.0.0.1:27017/bgexplorer"
        mongoengine.connect(host=app.config.get('SIMDB_URI', defaulturi))
        #                    alias="hiteffdb")

    def genqueries(self, request, findnewdata=True):
        hits = self.dbconfig.find_entries(source=request.spec.name,
                                          location=request.simvolume,
                                          distribution=request.spec.distribution)
        result = [request.clone(dataset=hit.id, query=hit.id,
                                livetime=hit.get_livetime(request.emissionrate))
                  for hit in hits]
        return result

    def evaluate(self, values, matches):
        """ Accept four values for 'values':
        None or empty list: do all the 'regular' values, in a dict
        'spectra': do all spectra at once, in a dict
        other string: do a single spectrum
        list of string: selection of spectra

        returns a dict with the result. Probably needs to be re-thought out
        """
        matches = matches if isinstance(matches, (list, tuple)) else [matches]
        if isinstance(values, str) and values != "spectra":
            values = [values]
        result = {}
        getspectra = (values is not None)
        for match in matches:
            if match.dataset is None:
                continue
            data = self.dbconfig.get_entry(match.dataset, getspectra)
            if data is None:
                continue
            multiplier = match.emissionrate if data.norm is not NormMultiplier.none else 1
            if values is None or len(values) == 0:
                for key, val in data.values.items():
                    result[key] = addnone(val*multiplier, result.get(key))
            elif values == 'spectra':
                for key, val in data.spectra.items():
                    result[key] = addnone(val*multiplier, result.get(key))
            elif isinstance(values, list):
                for key in values:
                    result[key] = addnone(data.spectra.get(key)*multiplier,
                                          result.get(key))
        return result


    def getdatasetdetails(self, datasetid, raw: bool = False):
        return self.dbconfig.get_entry(datasetid, includespectra=False,
                                       pymongo=raw)

    @property
    def collection(self):
        return self.dbconfig._get_db()[self.dbconfig.collection_name]

    def runquery(self, query, projection=None, sort=None):
        pipeline = []
        if query:
            # allow for lazy querying
            if not isinstance(query, dict):
                try:
                    query = bson.ObjectId(query)
                except bson.errors.InvalidId:
                    pass
                query = {'_id': query}
            pipeline.append({'$match': query})
        if sort:
            pipeline.append({'$sort': sort})
        if projection:
            pipeline.append({'$project': projection})
        return self.collection.aggregate(pipeline)


    def addentry(self, entry, fmt=''):
        """ Insert a new entry into the database
        Args:
            entry (dict or str): representation of database document to insert
            fmt (str): only 'json' is currently supported
        Returns:
            key: the _id for the newly inserted entry
        Raises:
            NotImplementedError: if fmt is not 'json'
            ValidationError: if entry is not a valid HitEfficiency
            pymongo error: if database insertion fails
        """
        if fmt.lower() == 'json':
            entry = self.dbconfig.validate_json(entry)
        else:
            raise NotImplementedError("Unhandled format %s", fmt)

        result = self.dbconfig.save_entry(entry)
        return result.id


