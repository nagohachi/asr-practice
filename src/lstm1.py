import random
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

root_dir = Path.cwd().parent
input_dir = root_dir / "input"
csj_dir = input_dir / "csj"

# train data
df = pd.read_csv(csj_dir / "text.train.char.csv")


vocab_path = Path(csj_dir / "vocab.char")
vocab_df = pd.read_csv(vocab_path, header=None)
vocab = vocab_df[0].tolist()
vocab_dict = {char: i for i, char in enumerate(vocab)}


class CSJTrainData:
    def __init__(
        self,
        train_df: pd.DataFrame,
        use_samples: int = 5000,
        max_spec_len: int = 500,
        batch_size: int = 4,
    ) -> None:
        # 最初の use_samples 個だけ使う
        self.train_df = train_df[:use_samples]
        # spectrogram の長さに関してソートする
        self.__sort_samples()

        self.train_df = self.train_df[
            self.train_df["spec_length"] <= max_spec_len
        ].reset_index(drop=True)
        print(f"Using {len(self.train_df)} samples")
        self.spec_npy_path_list = self.train_df["spec_npy_path"].to_list()
        self.target_list = self.train_df["target"].to_list()
        self.batch_size = batch_size

    def __sort_samples(self) -> None:
        self.train_df["spec_length"] = self.train_df["spec_npy_path"].apply(
            lambda x: np.load(x, allow_pickle=True).shape[0]
        )
        self.train_df = self.train_df.sort_values(by="spec_length").reset_index(
            drop=True
        )

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
            target = torch.Tensor(
                [vocab_dict[char] for char in self.target_list[index].split()]
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


# ## Datasetの準備

# num_mel_bins
input_features_size = 80
# vocab_size
hidden_features_size = len(vocab_dict)
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

        self.fc = nn.Linear(n_hidden_features, n_output_features)

        hidden_dim = n_layers * (2 if bidirectional else 1)
        self.h_0 = torch.zeros(hidden_dim, batch_size, n_hidden_features).to("cuda")
        self.c_0 = torch.zeros(hidden_dim, batch_size, n_hidden_features).to("cuda")

    def forward(self, x: nn.utils.rnn.PackedSequence) -> nn.utils.rnn.PackedSequence:
        # x: (batch_size, seq_len, input_features)
        x, _ = self.rnn(x)
        # x: (batch_size, seq_len, hidden_features)
        return x


# ## Training
dataset = CSJTrainData(train_df=df, batch_size=batch_size, use_samples=10000)

rnn = LSTMModel(
    n_input_features=input_features_size,
    n_output_features=len(vocab),
    n_hidden_features=len(vocab),
    bidirectional=True,
).to("cuda")


loss_fn = nn.CTCLoss(reduction="mean")
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-5)


train_loss_list = []


def train_loop(
    dataset: CSJTrainData,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 1,
    n_mean_loss: int = 1,
) -> None:
    training_data = []
    for X, y, X_length, y_length in dataset:
        training_data.append((X, y, X_length, y_length))

    # shuffle
    random.seed(0)
    random.shuffle(training_data)

    model.train()
    train_loss_list_tmp: list[float] = []
    for epoch in range(n_epochs):
        for batch, (X, y, X_length, y_length) in enumerate(tqdm(training_data)):
            # X, y を pad する
            X_pad = nn.utils.rnn.pad_sequence(X, batch_first=True)
            y_pad = nn.utils.rnn.pad_sequence(y, batch_first=True)
            # X を pack する
            X_packed = nn.utils.rnn.pack_padded_sequence(
                input=X_pad,
                lengths=torch.Tensor(X_length),
                batch_first=True,
                enforce_sorted=False,
            ).to("cuda")

            pred = model.forward(X_packed)
            pred, _ = nn.utils.rnn.pad_packed_sequence(pred, batch_first=True)
            pred = pred.log_softmax(2)

            # batch_first の状態から、CTCLoss に入力できるように変形する
            pred = pred.permute(1, 0, 2)

            loss = loss_fn(pred, y_pad, X_length, y_length)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch > 0 and batch % n_mean_loss == 0:
                train_loss_list.append(
                    sum(train_loss_list_tmp) / len(train_loss_list_tmp)
                )

                print(f"loss: {sum(train_loss_list_tmp) / len(train_loss_list_tmp)}")
                train_loss_list_tmp = []

            train_loss_list_tmp.append(loss.item())

        if train_loss_list_tmp:
            train_loss_list.append(sum(train_loss_list_tmp) / len(train_loss_list_tmp))
            print(f"loss: {sum(train_loss_list_tmp) / len(train_loss_list_tmp)}")
            train_loss_list_tmp = []


train_loop(
    dataset=dataset,
    model=rnn,
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=30,
    n_mean_loss=3,
)


plt.plot(train_loss_list)
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("train_loss.png")
