import pandas as pd
import pytest
from propen.match import match_data


@pytest.mark.parametrize(
    "heavy_light, property_th, same_epitope_threshold, expected_pairs",
    [
        pytest.param(
            1,
            0.5,
            10,
            {("ABCDEF", "ZZZTTT"), ("ABCDEF", "ABBDEF"), ("ABBDEF", "ZZZTTT")},
            id="low threshold high distance",
        ),
        pytest.param(
            1,
            0.5,
            2,
            {
                ("ABCDEF", "ABBDEF"),
            },
            id="low threshold low distance",
        ),
        pytest.param(1, 2, 10, {("ABCDEF", "ZZZTTT")}, id="high threshold high distance"),
    ],
)
def test_match_data(heavy_light, property_th, same_epitope_threshold, expected_pairs):
    property_name = "property"
    df = pd.DataFrame(
        data={
            "fv_heavy_aho": ["ABC", "ABB", "ZZZ"],
            "fv_light_aho": ["DEF", "DEF", "TTT"],
            property_name: [0, 1, 2],
        }
    )
    df["seq_col"] = df.apply(lambda x: x["fv_heavy_aho"] + x["fv_light_aho"], axis=1)

    df_matched = match_data(
        df,
        heavy_light=heavy_light,
        property_th=property_th,
        same_epitope_threshold=same_epitope_threshold,
        property_name=property_name,
    )
    pairs = [(s1, s2) for s1, s2 in zip(df_matched["s1"], df_matched["s2"])]
    assert set(pairs) == expected_pairs
