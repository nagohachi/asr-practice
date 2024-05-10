import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
import pandas as pd

load_dotenv()

root_dir = Path.cwd().parent
input_dir = root_dir / "input"
csj_dir = input_dir / "csj"
csj_train_dir = csj_dir / "train"
csj_eval_dir = csj_dir / "eval"
csj_train_dir.mkdir(parents=True, exist_ok=True)
csj_eval_dir.mkdir(parents=True, exist_ok=True)


def fetch_data_csj(csj_char_path: Path, csj_wav_path: Path) -> None:
    """加工済みの CSJ から

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


if __name__ == "__main__":
    CSJ_CHAR_PATH = os.environ.get("CSJ_CHAR_PATH")
    CSJ_WAV_PATH = os.environ.get("CSJ_WAV_PATH")

    assert CSJ_CHAR_PATH is not None
    assert CSJ_WAV_PATH is not None

    fetch_data_csj(csj_char_path=Path(CSJ_CHAR_PATH), csj_wav_path=Path(CSJ_WAV_PATH))
