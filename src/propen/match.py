import pandas as pd
from prescient.metrics.functional import edit_distance_pairwise


def match_data(
    df: pd.DataFrame,
    heavy_light=1,
    property_name="orcaa_hydrophobicity_negated",
    property_th=1,
    same_epitope_threshold=4,
) -> pd.DataFrame:
    """Input dataframe should only contain rows with the property available"""
    print("Running match_data")

    seq_col = "seq_col"
    expected_columns = [seq_col, property_name]

    for col in expected_columns:
        assert col in df.columns, f"Missing required column {col}"

    df = df.dropna(subset=[property_name]).reset_index(drop=True)

    edist = edit_distance_pairwise(df.seq_col)

    match_suffix = "_match"
    match_dfs = []
    columns = [seq_col, property_name]

    for i, row in df[columns].iterrows():
        # Create a dataframe with the matches for a given row
        # Keep only {columns}
        # Columns for the match have an added suffix
        edist_filter = (edist[i] > 0) * (edist[i] <= same_epitope_threshold)
        matches = df[columns].iloc[edist_filter]
        matches[[f"{col}{match_suffix}" for col in columns]] = tuple(row)
        matches["edist"] = edist[i][edist_filter]
        match_dfs.append(matches)

    df_matched = pd.concat(match_dfs)

    # df_matched = df_matched[
    #    ((df_matched[property_name + match_suffix] - df_matched[property_name]) >= property_th) & \
    #    ((df_matched[property_name + match_suffix] - df_matched[property_name]) <=3*property_th)
    # ]

    df_matched = df_matched[
        ((df_matched[property_name + match_suffix] - df_matched[property_name]) >= property_th)
    ]

    df_matched["s2"] = df_matched[seq_col + match_suffix]
    df_matched["s1"] = df_matched[seq_col]

    return df_matched.drop_duplicates([f"{col}", f"{col}{match_suffix}"])
