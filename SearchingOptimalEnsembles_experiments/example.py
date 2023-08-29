import logging

import SearchingOptimalEnsembles as SOE

logging.basicConfig(level=logging.DEBUG)

# SOE.run(metadataset_name="scikit-learn", surrogate_name="dre")
# SOE.run(metadataset_name="scikit-learn", surrogate_name="dkl")
# SOE.run(metadataset_name="quicktune", surrogate_name="dre")
SOE.run(metadataset_name="quicktune", surrogate_name="dkl")
