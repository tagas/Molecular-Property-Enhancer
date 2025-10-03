import argparse
import os
from typing import Union

import pandas as pd

from .match import match_data
from .sample import sample_propen
from .train_matched import train_propen
from .utils import preprocess_csv


def run_propen(
    input_file_or_dataframe: Union[str, pd.DataFrame] = "sequence_file.csv",
    seed: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    batch_size: int = 64,
    num_workers: int = 4,
    target: str = "UNK",
    model_name: str = "my_propen",
    heavy_light: int = 0,
    property: str = "affinity_pkd",
    property_th: float = 0.3,
    same_epitope_threshold: int = 10,
    max_iter: int = 5,
    n_samples: int = 15,
    n_seeds: int = 200,
    sample_cat: int = 1,
    temps: str = "1",
    output_dir: str = "results/",
    checkpoint: str = "",
    use_lr_scheduler: bool = False,
    store_intermediate_data: bool = True,
    tokenizer: str = "protein_seq_toks",
    preprocess: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    experiment_name : str, default="my_first_propen"
        name your run
    input_file : str, default="/gstore/home/tagasovn/axp_conditioning/c1.csv"
        A parquet file (or csv) with a column of aho ab sequences and a second column with
        values of property you wish to optimize.
    logger : str, default="wandb"
        Logger.
    gpus : int, default=1
        Number of GPUs used for training.
    seed : int, default=1
        seed
    lr : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay used by optimizer.
    epochs : int, default=300
        Number of epochs.
    batch_size : int, default=64
        Batch size.
    num_workers : int, default=4
        Number of workers.
    target : str, default="C1S"
        Target.
    model_name : str, default="propen_affinity"
        model name.
    heavy_light : int, default=0
        Match for edit distatance on heavy and light (1) or heavy chain only (0)
    property : str, default="affinity_pkd"
        Property name as stated in the input file
    property_th : float, default=0.3
        threshold for maximum difference in property value between s1 and s2:"
        " (s2_property - s1_property).
    same_epitope_threshold : int, default=10
        threshold for maximum edit distance between s1 and s2
    max_iter : int, default=5
        Number mutant iterations (at sampling time)
    n_samples : int, default=15
        Number of sampled mutants from the last layer (at sampling time)
    n_seeds : int, default=200
        Number of seeds to sample from.
    sample_cat : int, default=1
        Set to 1 if sampling from logits - last layer.
    temps : str, default="1"
        Comma separated list of temperatures."
        "1 is no temp scaling, any float >0.0 scales the multinomial sampling from the last layer
    model_dir : str, default="propen_models_test/"
        Where to save the trained PropEn.
    output_dir : str, default="results/"
        Where to store the results (can be in s3).
    checkpoint : str, default=""
        What checkpoint to use.
    use_lr_scheduler : bool, default=False
        Whether to use the learning rate scheduler.
    """

    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]
    else:
        output_dir = output_dir
    if output_dir.startswith("s3://"):
        print(f"Output is in S3: {output_dir}")
    else:
        print(f"Output dir is local: {output_dir}. Creating directories as needed.")
        os.makedirs(output_dir, exist_ok=True)

    if preprocess is not None:
        df, token_to_string, string_to_token = preprocess_csv(preprocess)
    else:
        # match data
        if isinstance(input_file_or_dataframe, pd.DataFrame):
            df = input_file_or_dataframe
        elif input_file_or_dataframe.endswith("csv"):
            df = pd.read_csv(input_file_or_dataframe)
        else:
            df = pd.read_parquet(input_file_or_dataframe)

    df = df.dropna(subset=property)
    print(df.shape)
    df_paired = match_data(
        df,
        heavy_light=heavy_light,
        property_name=property,
        property_th=property_th,
        same_epitope_threshold=same_epitope_threshold,
    )

    s1_max_len = df_paired["s1"].apply(len).max()
    s2_max_len = df_paired["s2"].apply(len).max()
    seq_len_adaptive = max(s1_max_len, s2_max_len)

    if store_intermediate_data:
        df_paired.to_parquet(f"{output_dir}/input_propen_train.parquet")
    print(
        f"Matching done, {len(df_paired)} pairs saved to {output_dir}/input_propen_train.parquet."
    )

    if checkpoint:
        # df_paired = pd.read_csv(input_file)
        print(f"Skipping training, using the passed checkpoint {checkpoint}")
        model_path = checkpoint
    else:
        # train propen
        model_path = train_propen(
            df_paired=df_paired,
            output_dir=output_dir,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            target=target,
            model_name=model_name,
            seed=seed,
            use_lr_scheduler=use_lr_scheduler,
            num_workers=num_workers,
            heavy_light=heavy_light,
            tokenizer=tokenizer,
        )
        print(f"Training done. Best checkpoint: {model_path}")

        if not model_path:
            raise ValueError("Training failed as no checkpoint was saved.")

    # sample from propen; currently set to start from training data.
    if not n_samples:
        print(f"Skipping sampling because n_samples = {n_samples}")
        print("Propen run complete")
        return
    else:
        list_of_df_designs = []

        print(f"Sampling for the following temperatures: {temps}")
        for temp in map(float, temps.split(",")):
            df_designs_temp = sample_propen(
                df_paired=df_paired,
                property_name=property,
                target=target,
                checkpoint=model_path,
                max_iter=max_iter,
                n_samples=n_samples,
                seed=seed,
                sample_cat=sample_cat,
                temp=temp,
                n_seeds=n_seeds,
                heavy_light=heavy_light,
                seq_len=seq_len_adaptive,
                tokenizer=tokenizer,
            )
            list_of_df_designs.append(df_designs_temp)

        df_designs_k = pd.concat(list_of_df_designs)

        if preprocess is not None:
            print(df_designs_k)
            df_designs_k["design_seq"] = df_designs_k["fv_heavy_light"].apply(
                lambda seq: [token_to_string[token] for token in seq]
            )
            df_designs_k["seed_seq"] = df_designs_k["s2"].apply(
                lambda seq: [token_to_string[token] for token in seq]
            )

        # Check whether the specified path exists or not
        os.makedirs(f"{output_dir}", exist_ok=True)
        output_path = output_dir + "/" + str(temp) + ".parquet"
        if store_intermediate_data:
            df_designs_k.to_parquet(output_path)
        print(output_path)

        print(f"Sampling done for temp={temp}.")
        print("Propen run complete")

        return df_designs_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="property enhancer training and inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_name", type=str, default="my_first_propen", help="name your run"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/gstore/home/tagasovn/axp_conditioning/c1.csv",
        help="A parquet file (or csv) with a column of aho ab sequences and a second column with"
        "values of property you wish to optimize",
    )
    parser.add_argument("--logger", type=str, default="wandb", help="Logger.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs used for training.")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay used by optimizer.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--target", type=str, default="C1S", help="Target.")
    parser.add_argument("--model_name", type=str, default="propen_affinity", help="model name.")
    parser.add_argument(
        "--heavy_light",
        type=int,
        default=0,
        help="Match for edit distatance on heavy and light (1) or heavy chain only (0)",
    )
    parser.add_argument(
        "--property",
        type=str,
        default="affinity_pkd",
        help="Property name as stated in the input file",
    )
    parser.add_argument(
        "--property_th",
        type=float,
        default=0.3,
        help="threshold for maximum difference in property value between s1 and s2:"
        " (s2_property - s1_property).",
    )
    parser.add_argument(
        "--same_epitope_threshold",
        type=int,
        default=10,
        help="threshold for maximum edit distance between s1 and s2",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5,
        help="Number mutant iterations (at sampling time)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=15,
        help="Number of sampled mutants from the last layer (at sampling time)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="protein_seq_toks",
        help="Choose the tokenizer accoring to your data type.",
    )

    parser.add_argument("--n_seeds", type=int, default=200, help="Number of seeds to sample from.")
    parser.add_argument(
        "--sample_cat",
        type=int,
        default=1,
        help="Set to 1 if sampling from logits - last layer.",
    )
    parser.add_argument(
        "--temps",
        type=str,
        default="1",
        help="Comma separated list of temperatures."
        "1 is no temp scaling, any float >0.0 scales the multinomial sampling from the last layer",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="propen_models_test/",
        help="Where to save the trained PropEn.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Where to store the results (can be in s3).",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default=None,
        help="Parquet file you want to use for training. Please include a column 'sequence'.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="What checkpoint to use.")
    parser.add_argument("--use_lr_scheduler", action="store_true")

    args = parser.parse_args()
    run_propen(**vars(args), store_intermediate_data=True)
