import hashlib
import io

import edlib
import Levenshtein
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.distributions.categorical import Categorical
from upath import UPath

from .constants import *
from .models import AE_matched
from .utils import Alphabet, read_toks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iterative_optimization(
    inp_seq_from_main, df_paired, seq_len, target, model, tokenizer, max_iter=3
):
    df_designs = pd.DataFrame([])
    edists = []

    # each mutant runs through optimization ###################
    for idx in np.arange(max_iter):
        if idx == 0:
            inp_seq = inp_seq_from_main
            current_seq = inp_seq
            initial_seq = inp_seq
        else:
            inp_seq = current_seq

        # alphabet = Alphabet(
        #    standard_toks=proteinseq_toks["toks"], prepend_bos=False, append_eos=False)

        if tokenizer == "custom_seq_toks":
            custom_seq_toks = read_toks()
            alphabet = Alphabet(
                standard_toks=custom_seq_toks["toks"], prepend_bos=False, append_eos=False
            )
            vocab_size = len(custom_seq_toks["toks"]) + 1
        else:
            alphabet = Alphabet(
                standard_toks=eval(tokenizer)["toks"], prepend_bos=False, append_eos=False
            )
            vocab_size = len(eval(tokenizer)["toks"]) + 1

        batch_converter = alphabet.get_batch_converter()
        _, _, X_in = batch_converter([("", current_seq)], max_len=seq_len)
        # seq_start      = inp_seq#df_paired.iloc[wt_idx]['s2']
        seqid_start = hashlib.md5(current_seq.encode()).hexdigest()

        Y_hat = torch.nn.functional.softmax(model(X_in), dim=-1)[0].detach().numpy()
        Y_hat = model(X_in).detach().numpy()
        out_seq_idx = model.onehot_enc.decode(torch.Tensor(Y_hat))
        out_seq_idx_aho = "".join(
            [alphabet.get_tok(i) for i in out_seq_idx[out_seq_idx != alphabet.padding_idx]]
        )

        # mutant sequence
        h_out = out_seq_idx_aho[:seq_len].replace("-", "")
        l_out = out_seq_idx_aho[seq_len:].replace("-", "")
        full_seq = h_out + l_out
        seq_mut = h_out + "*" + l_out
        seqid_mut = hashlib.md5(seq_mut.encode()).hexdigest()
        current_seq = out_seq_idx_aho  # full_seq

        reconstructed = model(X_in)
        reconstructed = torch.permute(reconstructed, (0, 2, 1))
        loss = model.loss(reconstructed, X_in[: model.seq_len])

        edist_tracking = edlib.align(inp_seq.replace("-", ""), full_seq)["editDistance"]
        edist_initial = edlib.align(initial_seq.replace("-", ""), full_seq)["editDistance"]
        edists.append(edist_tracking)
        if edist_tracking == 0:
            seq_type = "optim-fixed-p"
            break
        else:
            seq_type = "optim_" + str(idx)

        propen_pass = np.array(
            [
                h_out.replace("-", ""),
                l_out.replace("-", ""),
                full_seq,
                h_out,
                l_out,
                edist_initial,
                seq_type,
                seqid_start,
                seqid_mut,
                target,
                current_seq,
                loss.detach().numpy(),
            ]
        )
        propen_pass = propen_pass.reshape(-1, 12)
        df_propen_pass = pd.DataFrame(propen_pass)
        df_propen_pass.columns = [
            "fv_heavy",
            "fv_light",
            "fv_heavy_light",
            "fv_heavy_aho",
            "fv_light_aho",
            "edist_seed",
            "seq_type",
            "seed_id",
            "seq_id",
            "target",
            "s2",
            "reconst_loss",
        ]
        df_designs = pd.concat([df_designs, df_propen_pass])

        # inp_seq = current_seq
        # if edist_tracking == 0:
        #    break
    return df_designs


def sample_propen(
    df_paired: pd.DataFrame,
    property_name: str,
    target="",
    checkpoint="",
    max_iter=5,
    n_samples=5,
    seed=1,
    sample_cat=1,
    temp=1.0,
    n_seeds=200,
    heavy_light=1,
    test_all=False,
    seq_len=110,
    tokenizer="protein_seq_toks",
):
    pl.seed_everything(seed)

    if heavy_light:
        print("Sampling from both heavy and light")
        seq_len = 298

    if sample_cat == 0:
        temp = 0

    df_paired_original = df_paired.copy()
    # if seq exists in train, must have neighbours
    df_paired = df_paired.drop_duplicates(["s2"])

    if not test_all:
        # Pick the top 10% of binders
        df_paired = df_paired.sort_values(property_name, ascending=False)[:n_seeds]

    # alphabet = Alphabet(standard_toks=proteinseq_toks["toks"], prepend_bos=False, append_eos=False)
    if tokenizer == "custom_seq_toks":
        custom_seq_toks = read_toks()
        alphabet = Alphabet(
            standard_toks=custom_seq_toks["toks"], prepend_bos=False, append_eos=False
        )
        vocab_size = len(custom_seq_toks["toks"]) + 1
    else:
        alphabet = Alphabet(
            standard_toks=eval(tokenizer)["toks"], prepend_bos=False, append_eos=False
        )
        vocab_size = len(eval(tokenizer)["toks"]) + 1

    vocab_size = len(eval(tokenizer)["toks"]) + 1
    model = AE_matched(alphabet=alphabet, seq_len=seq_len, vocab_size=vocab_size)
    # chpt_path = checkpoint

    # for filename in os.listdir(chpt_path):
    #     cp = os.path.join(filename)

    # state_dict = torch.load(chpt_path + cp, map_location=device)["state_dict"]
    # support s3 checkpoints
    with UPath(checkpoint).open("rb") as f:
        buffer = io.BytesIO(f.read())
    state_dict = torch.load(buffer, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    df_designs = pd.DataFrame([])
    edists = []

    # sample mutants ##########################################
    # each mutant runs through optimization ###################

    for wt_idx in np.arange(df_paired.shape[0]):
        for idx in np.arange(max_iter):
            if idx == 0:
                inp_seq = df_paired.iloc[wt_idx]["s2"]
                current_seq = inp_seq
                initial_seq = inp_seq
            else:
                inp_seq = current_seq

            batch_converter = alphabet.get_batch_converter()
            _, _, X_in = batch_converter([("", current_seq)], max_len=seq_len)
            seq_start = df_paired.iloc[wt_idx]["s2"]
            seqid_start = hashlib.md5(seq_start.encode()).hexdigest()

            # Y_hat = torch.nn.functional.softmax(model(X_in), dim=-1)[0].detach().numpy()
            Y_hat = model(X_in)  # .detach().numpy()
            out_seq_idx = model.onehot_enc.decode(torch.Tensor(Y_hat))
            out_seq_idx_aho = "".join(
                [alphabet.get_tok(i) for i in out_seq_idx[out_seq_idx != alphabet.padding_idx]]
            )

            # mutant sequence
            h_out = out_seq_idx_aho[:seq_len].replace("-", "")
            l_out = out_seq_idx_aho[seq_len:].replace("-", "")
            full_seq = h_out + l_out
            seq_mut = h_out + "*" + l_out
            seqid_mut = hashlib.md5(seq_mut.encode()).hexdigest()
            current_seq = out_seq_idx_aho  # full_seq

            # reconstructed = torch.Tensor(Y_hat)
            reconstructed = torch.permute(Y_hat, (0, 2, 1))
            loss = model.loss(reconstructed, X_in[: model.seq_len])

            distance = Levenshtein.distance(full_seq, seq_start)
            edist_tracking = edlib.align(inp_seq.replace("-", ""), full_seq)["editDistance"]
            edist_initial = edlib.align(initial_seq.replace("-", ""), full_seq)["editDistance"]
            edists.append(edist_tracking)
            if edist_tracking == 0:
                seq_type = "mutant-fixed-p"
            else:
                seq_type = "mutant_" + str(idx)

            propen_pass = np.array(
                [
                    h_out,
                    l_out,
                    full_seq,
                    out_seq_idx_aho[:seq_len],
                    out_seq_idx_aho[seq_len:],
                    edist_initial,
                    seq_type,
                    seqid_start,
                    seqid_mut,
                    target,
                    seq_start,
                    loss.detach().numpy(),
                ]
            )
            propen_pass = propen_pass.reshape(-1, 12)
            df_propen_pass = pd.DataFrame(propen_pass)
            df_propen_pass.columns = [
                "fv_heavy",
                "fv_light",
                "fv_heavy_light",
                "fv_heavy_aho",
                "fv_light_aho",
                "edist_seed",
                "seq_type",
                "seed_id",
                "seq_id",
                "target",
                "s2",
                "reconst_loss",
            ]
            df_designs = pd.concat([df_designs, df_propen_pass])
            if edist_tracking == 0:
                break

    if sample_cat:
        # sample form last layer Categorical distribution ########################################
        for wt_idx in np.arange(df_paired.shape[0]):
            if wt_idx == 0:
                inp_seq = df_paired.iloc[wt_idx]["s2"]
                current_seq = inp_seq
                initial_seq = inp_seq
            else:
                inp_seq = current_seq

            batch_converter = alphabet.get_batch_converter()
            _, _, X_in = batch_converter([("", inp_seq)], max_len=seq_len)
            seq_start = df_paired.iloc[wt_idx]["s2"]
            seqid_start = hashlib.md5(seq_start.encode()).hexdigest()

            m = Categorical(logits=torch.Tensor(model(X_in)) / temp)
            for idx in np.arange(n_samples):
                sampled_s = m.sample()
                sampled_s = "".join(
                    [alphabet.get_tok(i) for i in sampled_s[sampled_s != alphabet.padding_idx]]
                )
                # mutant sequence
                h_out = sampled_s[:seq_len].replace("-", "")
                l_out = sampled_s[seq_len:].replace("-", "")
                full_seq = h_out + l_out
                seq_mut = h_out + "*" + l_out
                seqid_mut = hashlib.md5(seq_mut.encode()).hexdigest()
                current_seq = sampled_s  # full_seq

                edist_tracking = edlib.align(inp_seq.replace("-", ""), full_seq)["editDistance"]
                edist_initial = edlib.align(initial_seq.replace("-", ""), full_seq)["editDistance"]
                edists.append(edist_tracking)
                seq_type = "sampled_mutant_" + str(idx)

                # reconstructed = model(X_in)

                _, _, X_sampled = batch_converter([("", sampled_s)], max_len=seq_len)
                # X_sampled = torch.permute(X_sampled, (0, 2, 1))
                # X_sampled = torch.nn.functional.softmax(X_sampled.float(), dim=-1)[0]
                loss = (X_sampled.float() - X_in.float()).abs().mean()

                propen_pass = np.array(
                    [
                        h_out.replace("-", ""),
                        l_out.replace("-", ""),
                        full_seq.replace("-", ""),
                        sampled_s[:seq_len],
                        sampled_s[seq_len:],
                        edist_initial,
                        seq_type,
                        seqid_start,
                        seqid_mut,
                        target,
                        inp_seq,
                        loss.detach().numpy(),
                    ]
                )
                propen_pass = propen_pass.reshape(-1, 12)
                df_propen_pass = pd.DataFrame(propen_pass)
                df_propen_pass.columns = [
                    "fv_heavy",
                    "fv_light",
                    "fv_heavy_light",
                    "fv_heavy_aho",
                    "fv_light_aho",
                    "edist_seed",
                    "seq_type",
                    "seed_id",
                    "seq_id",
                    "target",
                    "s2",
                    "reconst_loss",
                ]

                for j in np.arange(df_propen_pass.shape[0]):
                    df_optimized = iterative_optimization(
                        df_propen_pass.iloc[j].fv_heavy_aho + df_propen_pass.iloc[j].fv_light_aho,
                        df_paired,
                        seq_len,
                        target,
                        model,
                        tokenizer,
                        max_iter,
                    )
                    df_propen_pass = pd.concat([df_propen_pass, df_optimized])

                df_designs = pd.concat([df_designs, df_propen_pass])

                # if edist_tracking == 0:
                #    break
            if wt_idx == 100:
                break
    df_designs_f = df_designs.drop_duplicates(["fv_heavy_aho", "fv_light_aho"])
    keep = []
    df_paired = df_paired_original
    # check for points from train
    for i in np.arange(df_designs_f.shape[0]):
        df_tmp = df_paired[
            (df_paired.s2.replace("-", "") == df_designs_f.iloc[i].fv_heavy_light)
            | (df_paired.s2.replace("-", "") == df_designs_f.iloc[i].fv_heavy_light)
        ]
        if df_tmp.shape[0] == 0:
            keep.append(i)

    df_designs_k = df_designs_f.iloc[keep]

    df_designs_k["sampling_temp"] = np.repeat(temp, df_designs_k.shape[0])
    df_designs_k["method"] = np.repeat("PropEn", df_designs_k.shape[0])
    df_designs_k.reset_index()
    print(df_designs_k.shape)
    print("# Sampling done ###############")
    return df_designs_k
