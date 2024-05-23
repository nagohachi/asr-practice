import random
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm

root_dir = Path.cwd().parent
input_dir = root_dir / "input"
csj_dir = input_dir / "csj"
model_dir = Path.cwd().parent / "models"


# train data
print("Loading data...")
df = pl.read_csv(csj_dir / "text.train.char.csv")


vocab_path = Path(csj_dir / "vocab.char")
vocab = []
with open(vocab_path, "r") as f:
    for line in f:
        vocab.append(line.strip())

# <blank> トークンを追加
vocab.append("<blank>")

print(f"Using {len(vocab)} characters")

vocab_to_id_dict = {char: i for i, char in enumerate(vocab)}
id_to_vocab_dict = {i: char for i, char in enumerate(vocab)}

sos_token_id = vocab_to_id_dict["<sos>"]
eos_token_id = vocab_to_id_dict["<eos>"]
blank_token_id = vocab_to_id_dict["<blank>"]

x_padding_value = 0
y_padding_value = blank_token_id


class CSJTrainData:
    def __init__(
        self,
        train_df: pl.DataFrame,
        use_samples: int = 5000,
        max_spec_len: int = 500,
        batch_size: int = 4,
    ) -> None:
        # 最初の use_samples 個だけ使う
        self.train_df = train_df[:use_samples]
        # spectrogram の長さに関してソートする
        print("Start sorting...")
        self.__sort_samples()
        print("Start filtering...")
        self.train_df = self.train_df.filter(pl.col("spec_length") <= max_spec_len)
        print(f"Using {len(self.train_df)} samples")
        self.spec_npy_path_list = self.train_df["spec_npy_path"].to_list()
        self.target_list = self.train_df["target"].to_list()
        self.batch_size = batch_size

    def __sort_samples(self) -> None:
        self.train_df = self.train_df.sort(by="spec_length")

    def __len__(self) -> int:
        return len(self.spec_npy_path_list) // self.batch_size

    def __getitem__(
        self, idx: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
        """先頭から batch_size 個だけ取ってリストにして返す

        Args:
            idx (int): インデックス

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
            (スペクトログラムのリスト, ターゲットのリスト, スペクトログラムの長さのリスト, ターゲットの長さのリスト)
        """
        specs = []
        targets = []
        spec_lengths = []
        target_lengths = []
        for i in range(self.batch_size):
            index = idx * self.batch_size + i
            if index >= len(self.spec_npy_path_list):
                break
            spec = torch.Tensor(
                np.load(self.spec_npy_path_list[index], allow_pickle=True)
            )
            spec = (spec - spec.mean(dim=0)) / spec.std(dim=0)
            spec_length = spec.size(0)

            # ターゲットの先頭に <sos> トークン、末尾に <eos> トークンを追加
            target = torch.Tensor(
                [sos_token_id]
                + [vocab_to_id_dict[char] for char in self.target_list[index].split()]
                + [eos_token_id]
            )
            target_length = target.size(0)
            specs.append(spec)
            targets.append(target)
            spec_lengths.append(spec_length)
            target_lengths.append(target_length)
        return specs, targets, spec_lengths, target_lengths

    def __iter__(
        self,
    ) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]]:
        for i in range(len(self)):
            yield self[i]


# ## LSTMModel のパラメータ
# num_mel_bins
input_features_size = 80
hidden_features_size = 320
# vocab_size
output_features_size = len(vocab)
bidirectional = True
n_layers = 1
batch_size = 10


# ## Model


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_features: int,
        n_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_input_features,
            hidden_size=n_hidden_features,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        fc_input_size = n_hidden_features * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_size, n_output_features)

        hidden_dim = n_layers * (2 if bidirectional else 1)
        self.h_0 = torch.zeros(hidden_dim, batch_size, n_hidden_features).to("cuda")
        self.c_0 = torch.zeros(hidden_dim, batch_size, n_hidden_features).to("cuda")

    def forward(self, x: nn.utils.rnn.PackedSequence) -> torch.Tensor:
        # x: (batch_size, seq_len, input_features)
        x, _ = self.rnn(x)
        x_tensor, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=blank_token_id
        )
        # x_tensor: (batch_size, seq_len, hidden_features)

        x_tensor = self.fc(x_tensor)
        # x_tensor: (batch_size, seq_len, output_features)

        # log_softmax
        x_tensor = x_tensor.log_softmax(dim=2)

        # batch_first の状態から、CTCLoss に入力できるように変形する
        x_tensor = x_tensor.permute(1, 0, 2)
        # x_tensor: (seq_len, batch_size, output_features)
        return x_tensor


# ## Training

dataset = CSJTrainData(train_df=df, batch_size=batch_size, use_samples=10000)

rnn = LSTMModel(
    n_input_features=input_features_size,
    n_output_features=output_features_size,
    n_hidden_features=hidden_features_size,
    n_layers=n_layers,
    bidirectional=bidirectional,
).to("cuda")

print(rnn)


loss_fn = nn.CTCLoss(reduction="mean")
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-5)


train_loss_list: list[float] = []

wandb.init(project="asr-practice", name="lstm1")


def train_loop(
    dataset: CSJTrainData,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 1,
) -> None:
    """訓練を行うための関数

    Args:
        dataset (CSJTrainData): 訓練データセット
        model (nn.Module): nn.LSTM のモデル
        loss_fn (nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        n_epochs (int, optional): 訓練で用いるエポック数. Defaults to 1.
    """

    def print_decoded_pred(pred: torch.Tensor) -> None:
        """モデルの出力をデコードして表示する

        Args:
            pred (torch.Tensor): モデルの出力
        """
        # batch_first の状態に戻す
        pred = pred.permute(1, 0, 2)
        # pred: (batch_size, seq_len, vocab_size)
        pred_argmax = pred.argmax(dim=2).cpu().numpy()
        # pred_argmax: (batch_size, seq_len)
        # バッチ内の i 番目の出力について
        for i in range(pred_argmax.shape[0]):
            pred_str = ""
            # i 番目の出力における j 番目の文字について
            for j in range(pred_argmax.shape[1]):
                # blank トークンは無視
                if pred_argmax[i, j] == blank_token_id:
                    continue
                # 連続する同じ文字は無視
                if j > 0 and pred_argmax[i, j] == pred_argmax[i, j - 1]:
                    continue
                pred_str += id_to_vocab_dict[pred_argmax[i, j]]
            print(pred_str)

    training_data = []
    print("Start preparing training data...")
    for X, y, X_length, y_length in tqdm(dataset, total=len(dataset)):
        training_data.append((X, y, X_length, y_length))

    # shuffle
    random.seed(0)
    random.shuffle(training_data)

    model.train()
    for epoch in range(n_epochs):
        train_loss_list_by_epoch: list[float] = []

        for batch, (X, y, X_length, y_length) in enumerate(tqdm(training_data)):
            # X, y を pad する
            X_pad = nn.utils.rnn.pad_sequence(
                X, batch_first=True, padding_value=blank_token_id
            )
            y_pad = nn.utils.rnn.pad_sequence(
                y, batch_first=True, padding_value=blank_token_id
            )
            # X を pack する
            X_packed = nn.utils.rnn.pack_padded_sequence(
                input=X_pad,
                lengths=torch.Tensor(X_length),
                batch_first=True,
                enforce_sorted=False,
            ).to("cuda")

            pred = model.forward(X_packed)

            # 試しに出力してみる
            print_decoded_pred(pred)

            loss = loss_fn(pred, y_pad, X_length, y_length)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train_loss": loss.item()})
            train_loss_list_by_epoch.append(loss.item())

        epoch_mean_loss = sum(train_loss_list_by_epoch) / len(train_loss_list_by_epoch)
        wandb.log({"train_epoch_loss": epoch_mean_loss})

        # save model
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / f"lstm1_epoch{epoch}.pth")


train_loop(
    dataset=dataset,
    model=rnn,
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=5,
)
