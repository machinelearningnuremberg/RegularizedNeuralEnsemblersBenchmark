from typing_extensions import Literal

from .metadatasets import MetaDatasetMapping


def run(
    metadataset_name: Literal["scikit-learn", "nasbench201", "quicktune"],
) -> None:
    metadataset = MetaDatasetMapping[metadataset_name]()
    dataset_name = "kr-vs-kp"
    hp_candidates = metadataset.get_hp_candidates(dataset_name)
    pipeline_hps, y, y_per_pipeline, t_per_pipeline = metadataset.get_batch(
        dataset_name=dataset_name, metric_name="nll"
    )

    print(
        f"hp_candidates.shape: {hp_candidates.shape}"
    )  # hp_candidates.shape: torch.Size([8533, 196])
    print(
        f"pipeline_hps.shape: {pipeline_hps.shape}"
    )  # pipeline_hps.shape: torch.Size([16, 6, 196])
    print(f"y.shape: {y.shape}")  # y.shape: torch.Size([16])
    print(
        f"y_per_pipeline.shape: {y_per_pipeline.shape}"
    )  # y_per_pipeline.shape: torch.Size([16, 6])
    print(
        f"t_per_pipeline.shape: {t_per_pipeline.shape}"
    )  # t_per_pipeline.shape: torch.Size([16, 6])
