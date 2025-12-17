import argparse
import json
import os

import laion_clap
import torch
from msclap import CLAP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import TTADataset


class TTAPreprocessDataset(TTADataset):
    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__(data_dir=data_dir, split=split)

    def __getitem__(self, idx: int) -> dict:
        data = self.database[idx]
        return data


class MSClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.model = CLAP(version="2023", use_cuda=True)
        self.max_text_len = 77  # MSCLAPのテキスト最大長
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """テキストをバッチで埋め込む"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embeddings(truncated_texts)
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """オーディオをバッチで埋め込む"""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embeddings(audio_files)
        return audio_embeddings.to(self.dtype)


class LaionClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt("models/630k-audioset-best.pt")
        self.max_text_len = 77  # LaionCLAPのテキスト最大長
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """テキストをバッチで埋め込む（長いテキストはトランケート）"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(
                truncated_texts, use_tensor=True
            )
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """オーディオをバッチで埋め込む"""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embedding_from_filelist(
                x=audio_files, use_tensor=True
            )
        return audio_embeddings.to(self.dtype)


class QwenParser:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
        }
        self.model.to(self.device)

    def _build_chat_template(self, text: str) -> str:
        """テキストからチャットテンプレートを構築する"""
        system_prompt = "You are a function that outputs ONLY valid JSON."
        user_prompt = f"""
            Extract the sound sources likely present in the audio caption below.

            Caption:
            {text}

            Return ONLY this JSON schema:
            {{["string"]}}
            """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _normalize_output(self, result_text: str) -> str:
        """出力テキストを正規化する（リスト形式に変換）"""
        result_text = result_text.strip()
        # 改行を除去して1行にする
        result_text = " ".join(result_text.split())
        # JSON配列部分を抽出（プレフィックスやサフィックスを除去）
        start_idx = result_text.find("[")
        end_idx = result_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx : end_idx + 1]
        elif not result_text.startswith("["):
            result_text = f"[{result_text}]"
        # クォートなしの文字列をクォートで囲む（例: [engine, steam] -> ["engine", "steam"]）
        result_text = self._fix_unquoted_strings(result_text)
        return result_text

    def _fix_unquoted_strings(self, result_text: str) -> str:
        """クォートなしの文字列をダブルクォートで囲む"""
        # 余分な { } ( ) を除去
        for char in "{}()":
            result_text = result_text.replace(char, "")
        # 既にパース可能ならそのまま返す
        try:
            json.loads(result_text)
            return result_text
        except json.JSONDecodeError:
            pass
        # [ と ] を除去して中身を取得
        inner = result_text[1:-1].strip()
        if not inner:
            return "[]"
        # カンマで分割し、各要素を処理
        items = []
        for item in inner.split(","):
            item = item.strip().strip('"').strip("'")
            # "key": "value" 形式の場合は value 部分のみ取得
            if ":" in item:
                item = item.split(":")[-1].strip().strip('"').strip("'")
            if item:
                items.append(item)
        quoted_items = [f'"{item}"' for item in items]
        return "[" + ", ".join(quoted_items) + "]"

    def _parse_json_result(self, result_text: str) -> list:
        """JSON文字列をパースしてリストに変換する"""
        try:
            result: list = json.loads(result_text)
            if isinstance(result, list):
                return result
            elif isinstance(result, str):
                return [result]
            elif isinstance(result, dict):
                return result["audio_sources"]
            else:
                raise ValueError("Unexpected JSON format")
        except Exception as e:
            raise RuntimeError(f"{e}: \n{result_text}")

    def _decode_single_response(self, ids: torch.Tensor, input_len: int) -> list:
        """単一の応答をデコードしてパースする"""
        result_text = self.tokenizer.decode(
            ids[input_len:],
            skip_special_tokens=True,
        )
        result_text = self._normalize_output(result_text)
        return self._parse_json_result(result_text)

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        chats = [self._build_chat_template(text) for text in texts]

        inputs = self.tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        ids = self.model.generate(**inputs, **self.generation_config)

        responses = []
        for i in range(len(chats)):
            result = self._decode_single_response(ids[i], len(inputs.input_ids[i]))
            responses.append(result)

        return responses


### feature saving functions ###


def save_feats(
    feats_dir: str,
    feats_name: str,
    dataset: str,
    file_name: str,
    feats: torch.Tensor,
) -> str:
    save_path = os.path.join(feats_dir, feats_name, dataset, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feats, save_path)
    return save_path


def save_batch_feats(
    feats_dir: str,
    datasets: list[str],
    feats_name: str,
    file_names: list[str],
    feats: torch.Tensor,
):
    """
    Save a batch of features to disk.
    Args:
        feats_dir: Directory to save features.
        datasets: List of dataset names corresponding to each audio file.
        feats_name: Name of the feature type (e.g., "msclap_audio"
        file_names: List of audio file names.
        feats: Tensor of features [B, D].
    Returns:
        None
    """
    for i in range(len(file_names)):
        save_feats(
            feats_dir=feats_dir,
            feats_name=feats_name,
            dataset=datasets[i],
            file_name=file_names[i],
            feats=feats[i],
        )


### feature extraction main ###


def msclap_extract(dataloader, feats_dir: str):
    msclap_embedder = MSClapEmbedder()

    for batch in tqdm(dataloader, desc="Extracting MSCLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in audio_files
        ]
        text_file_names = [f"{text_id}.pt" for text_id in text_ids]

        msclap_audio_embeddings = msclap_embedder.embed_audios(audio_files)
        msclap_text_embeddings = msclap_embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            "msclap_audio",
            audio_file_names,
            msclap_audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            "msclap_text",
            text_file_names,
            msclap_text_embeddings,
        )


def laionclap_extract(dataloader, feats_dir: str):
    laion_clap_embedder = LaionClapEmbedder()
    for batch in tqdm(dataloader, desc="Extracting LaionCLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in audio_files
        ]
        text_file_names = [f"{text_id}.pt" for text_id in text_ids]

        laion_audio_embeddings = laion_clap_embedder.embed_audios(audio_files)
        laion_text_embeddings = laion_clap_embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            "laionclap_audio",
            audio_file_names,
            laion_audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            "laionclap_text",
            text_file_names,
            laion_text_embeddings,
        )


def text_parse(dataloader, feats_dir: str):
    qwen_parser = QwenParser()
    for batch in tqdm(dataloader, desc="Parsing Text with Qwen"):
        texts = batch["text"]
        datasets = batch["dataset"]
        audio_files = batch["audio_file_path"]

        # Implement text parsing logic here using qwen_parser
        audio_sources: list[str] = qwen_parser.parse_texts(texts)

        for audio_file, dataset, sources in zip(audio_files, datasets, audio_sources):
            save_path = os.path.join(
                feats_dir, "parsed_texts", dataset, os.path.basename(audio_file)
            ).replace(".wav", ".json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(sources, f, ensure_ascii=False, indent=1)


def main(args):
    dataset = TTAPreprocessDataset(data_dir=args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    msclap_extract(dataloader, args.feats_dir)
    laionclap_extract(dataloader, args.feats_dir)
    # text_parse(dataloader, args.feats_dir)


### argument parser ###


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio Captioning Preprocessing")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--feats_dir",
        type=str,
        default="data/features",
        help="Directory to save the preprocessed features",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size for DataLoader",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    splits = ["train", "val", "test"]
    for split in splits:
        args.split = split
        main(args)
