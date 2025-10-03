import random
from dataclasses import asdict

import pandas as pd
import pytest
from omegaconf import OmegaConf

from propen.deployment.propen import DeploymentConfig, deployment_function


@pytest.fixture
def dataframe():
    n = 60

    base_fv_heavy_aho = "EVQLVES-GGGLVQPGGSLRLSCAASG-FNIKD-----TYIHWVRQAPGKGLEWVARIYPT---NGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWQIHG-------------------FYMMDYWGQGTLVTVSS"
    base_fv_light_aho = "DIQMTQSPSSLSASVGDRVT-ITCRA--SQDV------NTAVAWYQQKPGKAPKLLIY--------SASFLYSGVPSRFSGSRS--GTDFTLTISSLQPEDFATYYCQQYYK----------------------SPPTFGQGTKVEIK"

    return pd.DataFrame(
        {
            "fv_heavy_aho": [
                base_fv_heavy_aho + random.choice(("A", "R", "N", "D")) for _ in range(n)
            ],
            "fv_light_aho": [
                base_fv_light_aho + random.choice(("N", "T", "W", "Y")) for _ in range(n)
            ],
            "my_property": list(range(n)),
        }
    )


def test_propen(tmp_path, dataframe):
    output_columns = ["id"] + ["fv_heavy", "fv_light", "edist_seed"]

    config = DeploymentConfig(
        property="my_property",
        same_epitope_threshold=10,
        property_th=0,
        output_dirpath=str(tmp_path),
        model_name="propen_deployment_test",
        epochs=1,
        temps="1.0,1.5",
    )
    config = OmegaConf.create(asdict(config))

    out = deployment_function(dataframe=dataframe, config=config)

    assert isinstance(out, pd.DataFrame)
    assert all(col in out.columns for col in output_columns), (
        f"Expected columns: {output_columns} to be present but got {out.columns}."
    )
    assert len(out) > 10
    assert set(out["sampling_temp"]) == {1.0, 1.5}
