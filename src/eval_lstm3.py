import random
from pathlib import Path
from typing import Iterator

from jiwer import cer
import numpy as np
import polars as pl
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm


class CFG:
    input_features_size = 80
    hidden_features_size = 320
    bidirectional = True
    n_layers = 6
    batch_size = 20
    max_spec_len = 1500
    use_samples = 200000
    learning_rate = 5e-4
    loss_reduction = "mean"
    n_epochs = 10


root_dir = Path.cwd().parent
input_dir = root_dir / "input"
csj_dir = input_dir / "csj"
model_dir = Path.cwd().parent / "models"


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


class CSJEvalData:
    def __init__(
        self,
        eval_df: pl.DataFrame,
        batch_size: int = 4,
    ) -> None:
        self.eval_df = eval_df
        self.__sort_samples()
        print(f"Using {len(self.eval_df)} samples")
        self.spec_npy_path_list = self.eval_df["spec_npy_path"].to_list()
        self.target_list = self.eval_df["target"].to_list()
        self.batch_size = batch_size

    def __sort_samples(self) -> None:
        self.eval_df = self.eval_df.sort(by="spec_length")

    def __len__(self) -> int:
        return len(self.spec_npy_path_list) // self.batch_size

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[int], list[str]]:
        """先頭から batch_size 個だけ取ってリストにして返す

        Args:
            idx (int): インデックス

        Returns:
            tuple[list[torch.Tensor], list[int], list[str]:
            (スペクトログラムのリスト, スペクトログラムの長さのリスト, ターゲット)
        """
        specs = []
        targets = []
        spec_lengths = []
        for i in range(self.batch_size):
            index = idx * self.batch_size + i
            if index >= len(self.spec_npy_path_list):
                break
            spec = torch.Tensor(
                np.load(self.spec_npy_path_list[index], allow_pickle=True)
            )
            spec = (spec - spec.mean(dim=0)) / spec.std(dim=0)
            spec_length = spec.size(0)

            target = "".join(self.target_list[index].split())
            specs.append(spec)
            targets.append(target)
            spec_lengths.append(spec_length)
        return specs, spec_lengths, targets

    def __iter__(
        self,
    ) -> Iterator[tuple[list[torch.Tensor], list[int], list[str]]]:
        for i in range(len(self)):
            yield self[i]


# ## LSTMModel のパラメータ
# num_mel_bins
input_features_size = CFG.input_features_size
hidden_features_size = CFG.hidden_features_size
# vocab_size
output_features_size = len(vocab)
bidirectional = CFG.bidirectional
n_layers = CFG.n_layers
batch_size = CFG.batch_size


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

    def forward(
        self, x: nn.utils.rnn.PackedSequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_features)
        x, _ = self.rnn(x)
        # bidirectional なので、dim=2 で 2 つに分けて平均を取る
        x_tensor, x_lengths = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=x_padding_value
        )

        x_tensor = self.fc(x_tensor)
        # x_tensor: (batch_size, seq_len, output_features)

        # batch_first の状態から、CTCLoss に入力できるように変形する
        x_tensor = x_tensor.permute(1, 0, 2)
        # x_tensor: (seq_len, batch_size, output_features)

        # log_softmax
        x_tensor = x_tensor.log_softmax(dim=2)
        # x_tensor: (seq_len, batch_size, output_features)

        return x_tensor, x_lengths


# ## Evaling


rnn = LSTMModel(
    n_input_features=input_features_size,
    n_output_features=output_features_size,
    n_hidden_features=hidden_features_size,
    n_layers=n_layers,
    bidirectional=bidirectional,
)

rnn.load_state_dict(torch.load(model_dir / "lstm3_epoch9.pth"))

rnn = rnn.to("cuda")

print(rnn)


eval_loss_list: list[float] = []

# CFGオブジェクトのすべての属性を取得
cfg_vars = vars(CFG)

# JSONにシリアライズ可能な属性だけを選択
json_friendly_cfg_vars = {
    key: value
    for key, value in cfg_vars.items()
    if isinstance(value, (int, float, str, bool, list, dict, tuple, set))
}

wandb.init(project="asr-practice", name="lstm3", config=json_friendly_cfg_vars)


def eval_loop(
    dataset: CSJEvalData,
    model: nn.Module,
) -> None:
    """訓練を行うための関数

    Args:
        dataset (CSJEvalData): 訓練データセット
        model (nn.Module): nn.LSTM のモデル
    """

    def get_decoded_pred(pred: torch.Tensor) -> list[str]:
        res = []
        # batch_first の状態に戻す
        pred = pred.permute(1, 0, 2)
        # pred: (batch_size, seq_len, vocab_size)

        # vocab_size の最後の次元は blank なので、最後の次元をカット
        # pred = pred[:, :, :-1]

        # 最も確率の高いトークンを取得
        pred_argmax = pred.argmax(dim=2).cpu().numpy()
        # pred_argmax: (batch_size, seq_len)

        # バッチ内の各出力について
        for batch in pred_argmax:
            pred_str = ""
            # 出力における j 番目の文字について
            for j in range(batch.shape[0]):
                # blank トークンは無視
                if batch[j] == blank_token_id:
                    continue
                # 連続する同じ文字は無視
                if j > 0 and batch[j] == batch[j - 1]:
                    continue
                pred_str += id_to_vocab_dict[batch[j]]
            # もし pred_str に <sos> が含まれていれば、<sos> の次の文字から取り出す
            if "<sos>" in pred_str:
                pred_str = pred_str[pred_str.index("<sos>") + 5 :]
            # もし pred_str に <eos> が含まれていれば、<eos> の手前まで取り出す
            if "<eos>" in pred_str:
                pred_str = pred_str[: pred_str.index("<eos>")]
            res.append(pred_str)
        return res

    def print_decoded_pred(pred: torch.Tensor) -> None:
        """モデルの出力をデコードして表示する

        Args:
            pred (torch.Tensor): モデルの出力
        """
        decoded_pred = get_decoded_pred(pred)
        for pred_str in decoded_pred:
            print(pred_str)

    evaling_data = []
    print("Start preparing evaling data...")
    for X, X_length, y in tqdm(dataset, total=len(dataset)):
        evaling_data.append((X, X_length, y))

    # shuffle
    random.seed(0)
    random.shuffle(evaling_data)

    model.eval()
    eval_cer_list: list[float | dict] = []
    with torch.inference_mode():
        for batch, (X, X_length, y) in enumerate(tqdm(evaling_data)):
            # X, y を pad する
            X_pad = nn.utils.rnn.pad_sequence(
                X, batch_first=True, padding_value=x_padding_value
            )
            # X を pack する
            X_packed = nn.utils.rnn.pack_padded_sequence(
                input=X_pad,
                lengths=torch.Tensor(X_length),
                batch_first=True,
                enforce_sorted=False,
            ).to("cuda")

            pred, pred_lengths = model.forward(X_packed)

            # 試しに出力してみる
            pred_str = get_decoded_pred(pred)

            for p_str, t_str in zip(pred_str, y):
                # print(f"pred: {p_str}")
                # print(f"true: {t_str}")
                # print(f"CER: {cer(t_str, p_str)}")
                # print()
                eval_cer_list.append(cer(t_str, p_str))

    eval_cer: float = np.mean(eval_cer_list)  # type: ignore
    print(f"Mean CER: {eval_cer}")


for csv_path in csj_dir.glob("text.eval*.char.csv"):
    df = pl.read_csv(csv_path)
    print(csv_path)
    dataset = CSJEvalData(
        eval_df=df,
        batch_size=batch_size,
    )

    eval_loop(
        dataset=dataset,
        model=rnn,
    )
