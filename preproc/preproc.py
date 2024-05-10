import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from multiprocessing import Pool
from scipy.io import wavfile
from torch import Tensor
import torchaudio
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

root_dir = Path.cwd().parent
input_dir = root_dir / "input"

# CSJ
csj_dir = input_dir / "csj"
csj_raw_npy_dir = csj_dir / "raw_npy"
csj_spec_npy_dir = csj_dir / "spec_npy"

csj_raw_npy_dir.mkdir(exist_ok=True, parents=True)
csj_spec_npy_dir.mkdir(exist_ok=True, parents=True)

NUM_PROCESS_FOR_CONVERSION = 16
NUM_THREADS_FOR_CONVERSOIN = 16


def fetch_data_csj(csj_char_path: Path, csj_wav_path: Path) -> None:
    """加工済みの CSJ データsから train data, test data の CSV を作る

    Args:
        csj_char_path (Path): _description_
        csj_wav_path (Path): _description_
    """
    for text_file in csj_char_path.glob("*.char"):
        filename = text_file.name

        if "text" in filename:
            # text.train.char, text.eval1.char, text.eval2.char, text.eval3.char の場合
            # source と target が得られるので CSV に保存する

            # .wav までのパス
            source_path_list = []
            # target の文字列
            target_list = []

            # .wav のディレクトリ
            source_dir = (
                csj_wav_path / "wav.segments"
                if "train" in filename
                else csj_wav_path / "wav.segments.testset"
            )

            with open(text_file, "r") as f:
                for line in tqdm(f.readlines()):
                    # "A01M0097_0000211_0001695 え ー 内 容 と し ま し て は" みたいなのを split する
                    source_filename, target_text = line.split(maxsplit=1)
                    target_text = target_text.strip()

                    # {source_dir}/A01M0097/A01M0097_0000211_0001695.wav みたいなパスを作る
                    source_path = (
                        source_dir
                        / source_filename.split("_")[0]
                        / f"{source_filename}.wav"
                    )
                    source_path_list.append(source_path)
                    target_list.append(target_text)

            df = pd.DataFrame({"source_path": source_path_list, "target": target_list})
            df.to_csv(csj_dir / f"{filename}.csv", index=False)

        elif "vocab" in filename:
            # vocab.char の場合
            # csj_dir にそのままコピーして保存する
            with open(text_file, "r") as f:
                with open(csj_dir / filename, "w") as g:
                    g.write(f.read())


def convert_file_to_npy(row_output_dir_tuple: tuple[pd.Series, Path]) -> None:
    row, output_dir = row_output_dir_tuple
    source_path: Path = Path(row["source_path"])

    # ファイル名を作る
    filename = f"{source_path.stem}.npy"
    output_path = output_dir / filename

    # wav ファイルを読み込む
    rate, data = wavfile.read(source_path)

    # numpy 配列に変換して保存
    np.save(output_path, data)


def convert_wav_to_npy(df: pd.DataFrame, output_dir: Path) -> None:
    """CSJ の音声ファイルを numpy 配列に変換して保存する

    Args:
        df (pd.DataFrame): CSJ のデータが入った DataFrame
        output_dir (Path): 保存先のディレクトリ
    """
    with ThreadPoolExecutor(max_workers=NUM_THREADS_FOR_CONVERSOIN) as executor:
        list(
            tqdm(
                executor.map(
                    convert_file_to_npy,
                    [(row, output_dir) for _, row in df.iterrows()],
                ),
                total=df.shape[0],
            )
        )


def convert_file_to_spectrogram(row_output_dir_tuple: tuple[pd.Series, Path]) -> None:
    row, output_dir = row_output_dir_tuple
    source_path: Path = Path(row["source_path"])

    # ファイル名を作る
    filename = f"{source_path.stem}.npy"
    output_path = output_dir / filename

    # wav ファイルを読み込む
    rate, data = wavfile.read(source_path)

    # waveform (Tensor) – Tensor of audio of size (c, n) where c is in the range [0,2)
    waveform = Tensor(data).unsqueeze(0)
    spectrogram = torchaudio.compliance.kaldi.fbank(waveform=waveform, num_mel_bins=80)

    # numpy 配列に変換して保存
    np.save(output_path, spectrogram.numpy())


def convert_wav_to_spectrogram(df: pd.DataFrame, output_dir: Path) -> None:
    """音声ファイルをスペクトログラムに変換して保存する

    Args:
        npy_path (Path): 音声ファイルのパス
        output_dir (Path): 保存先のディレクトリ
    """
    with Pool(processes=NUM_PROCESS_FOR_CONVERSION) as p:
        _ = list(
            tqdm(
                p.imap(
                    convert_file_to_spectrogram,
                    [(row, output_dir) for _, row in df.iterrows()],
                ),
                total=df.shape[0],
            )
        )


if __name__ == "__main__":
    csj_csv_set = {
        "text.eval1.char.csv",
        "text.eval2.char.csv",
        "text.eval3.char.csv",
        "text.train.char.csv",
        "vocab.char",
    }

    # CSJ のデータがすべて揃っているか確認
    if not all((csj_dir / filename).exists() for filename in csj_csv_set):
        print("Creating CSV files for CSJ data")
        CSJ_CHAR_PATH = os.environ.get("CSJ_CHAR_PATH")
        CSJ_WAV_PATH = os.environ.get("CSJ_WAV_PATH")

        assert CSJ_CHAR_PATH is not None
        assert CSJ_WAV_PATH is not None

        fetch_data_csj(
            csj_char_path=Path(CSJ_CHAR_PATH), csj_wav_path=Path(CSJ_WAV_PATH)
        )

    # csj_npy_dir の中に .npy ファイルがなければ作る
    if not any(csj_raw_npy_dir.glob("*.npy")):
        print("Creating .npy files for CSJ data")
        for csv_file in csj_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            convert_wav_to_npy(df, csj_raw_npy_dir)

    if not any(csj_spec_npy_dir.glob("*.npy")):
        print("Creating spectrogram .npy files for CSJ data")
        for csv_file in csj_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            convert_wav_to_spectrogram(df, csj_spec_npy_dir)
