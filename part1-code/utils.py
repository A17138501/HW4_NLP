import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import string  # 新增
random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# ---------------------------
# 下面是自定义 typo transform
# ---------------------------

# 简单 QWERTY 邻居键映射，用来做更真实的打字错误
KEYBOARD_NEIGHBORS = {
    'q': 'w',
    'w': 'qe',
    'e': 'wr',
    'r': 'et',
    't': 'ry',
    'y': 'tu',
    'u': 'yi',
    'i': 'uo',
    'o': 'ip',
    'p': 'o',

    'a': 's',
    's': 'ad',
    'd': 'sf',
    'f': 'dg',
    'g': 'fh',
    'h': 'gj',
    'j': 'hk',
    'k': 'jl',
    'l': 'k',

    'z': 'x',
    'x': 'zc',
    'c': 'xv',
    'v': 'cb',
    'b': 'vn',
    'n': 'bm',
    'm': 'n',
}


def _is_alpha_token(token: str) -> bool:
    """
    判断一个 token 是否主要由字母构成，避免对纯标点/数字乱改。
    """
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    return len(letters) / max(len(token), 1) >= 0.6


def _swap_typo(token: str) -> str:
    """
    交换相邻两个字符，例如 movie -> mvoie
    """
    if len(token) < 4:
        return token
    i = random.randint(0, len(token) - 2)
    chars = list(token)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return ''.join(chars)


def _delete_typo(token: str) -> str:
    """
    删除一个字符，例如 great -> grat
    """
    if len(token) < 4:
        return token
    i = random.randint(0, len(token) - 1)
    chars = list(token)
    del chars[i]
    return ''.join(chars)


def _keyboard_sub_typo(token: str) -> str:
    """
    将一个字符替换成 QWERTY 键盘上的邻居键，例如 good -> gpod
    """
    indices = [i for i, ch in enumerate(token) if ch.lower() in KEYBOARD_NEIGHBORS]
    if not indices:
        return token
    i = random.choice(indices)
    ch = token[i]
    neighbors = KEYBOARD_NEIGHBORS.get(ch.lower(), '')
    if not neighbors:
        return token
    new_ch = random.choice(neighbors)
    if ch.isupper():
        new_ch = new_ch.upper()
    chars = list(token)
    chars[i] = new_ch
    return ''.join(chars)


def _corrupt_token(token: str, typo_prob: float = 0.3) -> str:
    """
    以一定概率对 token 施加一个 typo 操作。
    - 只对主要由字母组成且长度 >= 4 的 token 操作
    - typo_prob 控制整体噪声强度
    """
    if not _is_alpha_token(token):
        return token
    if len(token) < 4:
        return token
    if random.random() > typo_prob:
        return token

    op = random.choice(["swap", "delete", "sub"])
    if op == "swap":
        return _swap_typo(token)
    elif op == "delete":
        return _delete_typo(token)
    else:
        return _keyboard_sub_typo(token)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ####

    # 模拟用户真实打字错误的 transformation：
    # 1. 将句子按空格分词
    # 2. 对每个词，以 typo_prob 的概率应用一次 swap/delete/keyboard-sub 三种错误之一
    # 3. 将修改后的 token 重新拼回句子

    text = example["text"]
    tokens = text.split()
    transformed_tokens = [_corrupt_token(tok, typo_prob=0.3) for tok in tokens]
    example["text"] = " ".join(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
