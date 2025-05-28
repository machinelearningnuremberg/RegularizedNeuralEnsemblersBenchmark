import RegularizedNeuralEnsemblers.metadatasets.quicktune.metadataset as qmd

from ..posthoc.cmaes_ensembler import CMAESEnsembler

if __name__ == "__main__":
    DATA_DIR = "/work/dlclarge2/janowski-quicktune/predictions"
    metric_name = "error"
    data_version = "micro"
    task_id = 5
    metadataset = qmd.QuicktuneMetaDataset(
        data_dir=DATA_DIR, metric_name=metric_name, data_version=data_version
    )

    metadataset.set_state(metadataset.get_dataset_names()[task_id])
    cmaes = CMAESEnsembler(metadataset=metadataset)
    cmaes.sample([1, 2, 3])
