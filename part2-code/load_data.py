import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0  # t5-small 的 pad token id 就是 0

def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        """
        split: 'train', 'dev', or 'test'
        """
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        self.pad_id = self.tokenizer.pad_token_id
        # 直接用 pad_token_id 作为 decoder 的 BOS，和 T5 默认一致
        self.bos_id = self.pad_id

        (
            self.encoder_inputs,
            self.decoder_inputs,
            self.decoder_targets,
            self.initial_decoder_inputs,
        ) = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """
        返回四个 list:
            encoder_inputs: List[List[int]]
            decoder_inputs: List[List[int]] or None
            decoder_targets: List[List[int]] or None
            initial_decoder_inputs: List[List[int]]
        """
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        initial_decoder_inputs = []

        # train / dev 有 SQL target
        if split in ["train", "dev"]:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines)

            for nl, sql in zip(nl_lines, sql_lines):
                # encoder：加前缀
                enc_text = "translate English to SQL: " + nl.strip()
                enc_ids = tokenizer.encode(
                    enc_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=256,
                )

                # decoder target：SQL
                tgt_ids = tokenizer.encode(
                    sql.strip(),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=128,
                )

                # decoder input = BOS + target[:-1]
                if len(tgt_ids) > 0:
                    dec_in = [self.bos_id] + tgt_ids[:-1]
                    dec_tgt = tgt_ids
                else:
                    dec_in = [self.bos_id]
                    dec_tgt = [self.bos_id]

                encoder_inputs.append(enc_ids)
                decoder_inputs.append(dec_in)
                decoder_targets.append(dec_tgt)
                # initial_decoder_inputs 仅用于需要从单个 BOS 起步的情况，目前实际上没用到，但保留接口
                initial_decoder_inputs.append([self.bos_id])

        # test：只有 NL，没有 SQL
        else:  # split == "test"
            for nl in nl_lines:
                enc_text = "translate English to SQL: " + nl.strip()
                enc_ids = tokenizer.encode(
                    enc_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=256,
                )
                encoder_inputs.append(enc_ids)
            # test 时 initial_decoder_inputs 只在 generate 的时候用一个 BOS 起步（目前没真正用到）
            initial_decoder_inputs = [[self.bos_id] for _ in encoder_inputs]
            decoder_inputs = None
            decoder_targets = None

        return encoder_inputs, decoder_inputs, decoder_targets, initial_decoder_inputs

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        enc_ids = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)

        if self.split in ["train", "dev"]:
            dec_in = torch.tensor(self.decoder_inputs[idx], dtype=torch.long)
            dec_tgt = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
            init_dec = torch.tensor(self.initial_decoder_inputs[idx], dtype=torch.long)
            # 和 train loop 约定：返回 4 个，collate 再组装成 5 个
            return enc_ids, dec_in, dec_tgt, init_dec
        else:
            init_dec = torch.tensor(self.initial_decoder_inputs[idx], dtype=torch.long)
            return enc_ids, init_dec


def normal_collate_fn(batch):
    """
    train/dev:
      batch = [(enc_ids, dec_in, dec_tgt, init_dec), ...]
    返回:
      encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    """
    enc_ids_list, dec_in_list, dec_tgt_list, init_dec_list = zip(*batch)

    encoder_ids = pad_sequence(enc_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(dec_in_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(dec_tgt_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(init_dec_list, batch_first=True, padding_value=PAD_IDX)

    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    """
    test:
      batch = [(enc_ids, init_dec), ...]
    返回:
      encoder_ids, encoder_mask, initial_decoder_inputs
    """
    enc_ids_list, init_dec_list = zip(*batch)

    encoder_ids = pad_sequence(enc_ids_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(init_dec_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x

