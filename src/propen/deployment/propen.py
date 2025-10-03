from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig
from propen.run_propen import run_propen


@dataclass
class DeploymentConfig:
    property_th: float = 0.0
    same_epitope_threshold: int = 0
    property: str = ""
    output_dirpath: str = ""
    epochs: int = 300
    model_name: str = "propen_deployment"
    n_samples: int = 15
    temps: str = "1"
    heavy_light: int = 1
    max_iter: int = 5


def deployment_function(dataframe: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Trains PropEn and samples new designs.

    Parameters
    ----------
    df_seeds : pd.DataFrame
        Input dataframe with seeds:
            - ID column: id
            - fv_heavy_aho,fv_light_aho

            For generative models, the ID column is treated as a seed ID
            and can be duplicated in the output (when there is >1 generated design per input).

    config : DictConfig
        OmegaConf's DictConfig object with fields defined in DeploymentConfig

    Returns
    -------
    pd.DataFrame
        Output dataframe with:

            - seed id column: id (can be duplicated)
            - new columns: fv_heavy,fv_light,edist_seed

    """
    assert "fv_heavy_aho" in dataframe
    assert "fv_light_aho" in dataframe
    print(f"Running deployment function with config: {config}")

    if "seq_col" not in dataframe:
        if "heavy_light" == 1:
            dataframe["seq_col"] = dataframe.apply(
                lambda row: row["fv_heavy_aho"] + row["fv_light_aho"], axis=1
            )
        else:
            dataframe["seq_col"] = dataframe["fv_heavy_aho"].tolist()

    output_df = run_propen(
        input_file_or_dataframe=dataframe,
        property_th=config.property_th,
        property=config.property,
        same_epitope_threshold=config.same_epitope_threshold,
        output_dir=config.output_dirpath,
        epochs=config.epochs,
        store_intermediate_data=True,
        model_name=config.model_name,
        preprocess=None,
        n_samples=config.n_samples,
        temps=config.temps,
        heavy_light=config.heavy_light,
        max_iter=config.max_iter,
    )
    if output_df is None:
        raise ValueError("No new designs were generated")

    output_df["id"] = output_df["seed_id"]
    return output_df
