import SearchingOptimalEnsembles.metadatasets.synthetic.metadataset as syn_md

if __name__ == "__main__":
    metadataset = syn_md.SyntheticMetaDataset()
    metadataset.set_state("10")
    metadataset.evaluate_ensembles([[1, 2], [3, 4]])
    print("Done")
