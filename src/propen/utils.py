import itertools
import random
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils import data

import string
import itertools
import pandas as pd
import ast
import pyarrow.parquet as pq
import pyarrow as pa
import json

def read_toks():
    try:
        with open('constants.txt', 'r') as file:
            content = file.read()

            # Extract the dictionary part (find the last custom_seq_toks definition)
            start_idx = content.rfind("custom_seq_toks = {")
            if start_idx != -1:
                dictionary_part = content[start_idx:].split("=", 1)[1].strip()
                custom_seq_toks = ast.literal_eval(dictionary_part)
                
            else:
                print("custom_seq_toks not found in the file.")
    except SyntaxError as e:
        print(f"Error in parsing dictionary: {e}")
    except FileNotFoundError:
        print("File not found. Please ensure 'constants.txt' exists.")
        
    return custom_seq_toks




def generate_single_char_tokens():
    basic_chars = (
        ''.join(chr(i) for i in range(33, 127))  # Basic ASCII (printable)
        + ''.join(chr(i) for i in range(0x0370, 0x03FF))  # Greek
        + ''.join(chr(i) for i in range(0x0400, 0x04FF))  # Cyrillic
        + ''.join(chr(i) for i in range(0x0590, 0x05FF))  # Hebrew
        + ''.join(chr(i) for i in range(0x0600, 0x06FF))  # Arabic
        + ''.join(chr(i) for i in range(0x0900, 0x097F))  # Devanagari
    )
    for char in basic_chars:
        yield char

def preprocess_csv(file_path='PD_Nrf2_processed.parquet'):

    mcp_df = pd.read_parquet(file_path)
    #mcp_df['sequence'] = mcp_df['sequence'].apply(lambda x: str(ast.literal_eval(x.decode('utf-8'))) if isinstance(x, bytes) else ast.literal_eval(x))

    # Generate unique strings from sequences
    unique_strings = set(item for sublist in mcp_df['sequence'] for item in sublist)
    
    # Create a token generator
    token_generator = generate_single_char_tokens()

    # Ensure enough tokens for unique strings
    required_tokens = len(unique_strings)
    available_tokens = sum(1 for _ in generate_single_char_tokens())
    if required_tokens > available_tokens:
        raise ValueError(f"Not enough single-character tokens! Required: {required_tokens}, Available: {available_tokens}")

    # Map unique strings to tokens
    string_to_token = {string: next(token_generator) for string in unique_strings}
    token_to_string = {token: string for string, token in string_to_token.items()}

    # Tokenize the sequences
    mcp_df['tokenized'] = mcp_df['sequence'].apply(lambda seq: [string_to_token[s] for s in seq])
    mcp_df['seq_col'] = mcp_df['sequence']#.apply(lambda seq: [string_to_token[s] for s in seq])

    # Detokenize the sequences
    mcp_df['detokenized'] = mcp_df['tokenized'].apply(lambda seq: [token_to_string[token] for token in seq])
    unique_tokens = set(item for sublist in mcp_df['seq_col'] for item in sublist)

    with open('constants.txt', "w") as file:
        file.write("\ncustom_seq_toks = {\n    'toks': [\n")
        for item in unique_tokens:
            # Use ascii() to escape special characters
            file.write(f"        {ascii(item)},\n")
        file.write("    ]\n}\n")

    return mcp_df, token_to_string, string_to_token

    
    

eps = 1e-4


def entropy(p):
    return -(torch.log(p + eps) * p).sum(axis=1)


def ll(p):
    return (torch.log(p + eps)).sum(axis=1)


class PaddCollator(object):
    def __init__(self, alphabet, max_len=None):
        self.alphabet = alphabet
        self.max_len = max_len
        self.batch_converter = self.alphabet.get_batch_converter()

    def __call__(self, batch):
        if len(batch[0]) != 2:
            samples = [("", seq) for seq in batch]
            _, _, HL = self.batch_converter(samples, max_len=self.max_len)

        else:
            h_samples = [("", seq[0]) for seq in batch]
            l_samples = [("", seq[1]) for seq in batch]

            _, _, H = self.batch_converter(h_samples, max_len=self.max_len)
            _, _, L = self.batch_converter(l_samples, max_len=self.max_len)
            HL = (H, L)

        return HL


class DatasetPPI_matched(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, col_names: list = ["s1", "s2"]):
        super().__init__()
        self.df = dataframe
        self.col_names = col_names

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Select sample
        row = self.df.iloc[index]
        if len(self.col_names) == 2:
            seq1 = "".join([row[self.col_names[0]]])
            seq2 = "".join([row[self.col_names[1]]])

        else:
            assert False
        return seq1, seq2


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]], max_len=None):
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]

        if max_len is None:
            # find max len in a batch
            max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)

        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )

        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            # print(len(seq_encoded_list[0]))
            # ed = edlib.align(seq_encoded_list[0], seq_encoded_list[1])["editDistance"]
            # labels.append(ed)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.bos_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded) + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        # print(labels)
        return labels, strs, tokens


# based on: https://github.com/facebookresearch/esm
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        mask_toks: Sequence[str] = ("<mask>",),
        prepend_toks: Sequence[str] = ("<bos>",),
        append_toks: Sequence[str] = ("<eos>",),
        prepend_bos: bool = False,
        append_eos: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.mask_toks = list(mask_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.standard_toks)
        self.all_toks.extend(self.mask_toks)
        if self.prepend_bos:
            self.all_toks.extend(self.prepend_toks)
        if self.append_eos:
            self.all_toks.extend(self.append_toks)
        self.unique_no_split_tokens = self.all_toks

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.padding_idx = self.get_idx("<mask>")
        if self.prepend_bos:
            self.bos_idx = self.get_idx("<bos>")
        if self.append_eos:
            self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self.all_toks)

    def get_rnd_seq(self, size, seq_len):
        x = torch.randint(0, len(self.standard_toks), (size, seq_len))
        if self.prepend_bos:
            x[:, 0] = self.bos_idx
        if self.append_eos:
            x[:, -1] = self.eos_idx
        return x

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        return BatchConverter(self)

    def _tokenize(self, text) -> str:
        return text.split()

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.
        Args:
            text (:obj:`str`):
                The sequence to be encoded.
        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)

            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)

        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exists(x):
    return x is not None



import edlib
import numpy
from scipy.spatial.distance import pdist, squareform

#from prescient.metrics.utils import check_sequences_references_length


def _edit_distance(sequence1: str, sequence2: str) -> int:
    """Compute the edit distance between two sequences using edlib library."""

    if sequence1 is None or sequence2 is None:
        return numpy.nan

    output = edlib.align(sequence1, sequence2, task="distance")
    return output["editDistance"]


#@check_sequences_references_length
def edit_distance(sequences: list[str], references: list[str]) -> numpy.ndarray:
    """
    Compute the edit distance between a list of sequences and a list of references.
    This function checks if the number of sequences and references are the same before computing.

    Parameters
    ----------
    sequences : list[str]
        The list of sequences.
    references : list[str]
        The list of references.

    Returns
    -------
    numpy.ndarray
        An array of edit distances between the sequences and references.
    """
    return numpy.array(
        [
            _edit_distance(sequence, reference)
            for sequence, reference in zip(sequences, references)
        ]
    )


def edit_distance_pairwise(sequences: list[str]) -> numpy.ndarray:
    """
    Compute the pairwise edit distance between a list of sequences using scipy library.

    Parameters
    ----------
    sequences : list[str]
        The list of `n` sequences.

    Returns
    -------
    numpy.ndarray
        The pairwise edit distances with shape: `(n, n)`
    """
    # Reshape is required for pdist
    sequences = numpy.array(sequences).reshape(-1, 1)
    d = pdist(sequences, metric=lambda x, y: _edit_distance(x[0], y[0]))
    return squareform(d)


def edit_distance_one_to_many(sequences: str, references: list[str]) -> numpy.ndarray:
    """
    For each sequence in `sequences`, compute the edit distance to each sequence in `references`.
    Contrary to `edit_distance` function, the number of samples in `sequences` and `references`
    does not have to be the same since they are not treated as pairs.

    The result is a 2D numpy array of shape: $$(n_{sequences}, n_{references})$$.


    Example

    ```py
    sequences = ["abc", "abd", "abcd", ""]
    references = ["abc", "ad"]

    edit_distance_one_to_many(sequences, references)
    >>> array([[0., 2.],
       [1., 1.],
       [1., 2.],
       [3., 2.]])

    ```

    Parameters
    ----------
    sequences : str
        A list of sequences
    references : list[str]
        The list of references.

    Returns
    -------
    numpy.ndarray
        An array of edit distances of each sequences to references.
        Shape: (n_{sequences}, n_{references}).
    """
    result = numpy.empty(shape=(len(sequences), len(references)))

    for i in range(len(sequences)):
        sequence = sequences[i]
        result[i, :] = numpy.array(
            [_edit_distance(sequence, reference) for reference in references]
        )

    return result
