import argparse
import json
import os
import re

import laion_clap
import torch
from msclap import CLAP

# from sam_audio import SAMAudio, SAMAudioProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import TTADataset
from fd_openl3.openl3_fd import extract_embeddings as openl3_extract_embeddings
from utils.helper_func import fix_seed

### laion clap fix ###
_torch_load = torch.load


def torch_load_no_wo(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = torch_load_no_wo


class TTAPreprocessDataset(TTADataset):
    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__(data_dir=data_dir, split=split)

    def __getitem__(self, idx: int) -> dict:
        data = self.database[idx]
        return data


class MSClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32, seed: int = 42):
        self.model = CLAP(version="2023", use_cuda=True)
        self.max_text_len = 77  # MSCLAP max text length
        self.dtype = dtype
        self.seed = seed

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embeddings(truncated_texts)
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """Embed audios in batch"""
        fix_seed(self.seed)  # Ensure seed is fixed before audio embedding
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embeddings(audio_files)
        return audio_embeddings.to(self.dtype)


class LaionClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt("models/630k-audioset-best.pt")
        self.model.eval()
        self.max_text_len = 77  # LaionCLAP max text length
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch (truncate long texts)"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(
                truncated_texts, use_tensor=True
            )
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """Embed audios in batch"""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embedding_from_filelist(
                x=audio_files, use_tensor=True
            )
        return audio_embeddings.to(self.dtype)


class Openl3Embedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def embed_audios(
        self,
        audio_files: list[str],
        samplingrate: int = 16000,
        content_type: str = "music",
    ) -> torch.Tensor:
        """Embed audios in batch"""
        batch_size = len(audio_files)
        audio_embeddings = openl3_extract_embeddings(
            directory_path=audio_files[0],
            channels=1,
            samplingrate=samplingrate,
            content_type=content_type,
            openl3_hop_size=0.5,
            batch_size=batch_size,
        )
        audio_embeddings = torch.tensor(audio_embeddings)
        if batch_size == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)
        return audio_embeddings.to(self.dtype)


class QwenTextParser:
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
        """Build chat template from text"""
        system_prompt = "Output only a JSON array of strings."
        user_prompt = f"""Split the caption into a list of distinct sound events.
Preserve all modifiers (adjectives, adverbs, and descriptive phrases).
Do NOT include any temporal or sequential information (e.g., order, timing, repetition, before/after).
Do NOT output duplicate or semantically overlapping sound events.
If the same sound event appears multiple times, keep only the most informative occurrence.

Caption:
{text}

Example:
Caption: Birds chirp loudly in the distance; a person talks nearby; more chirping.
Output: ["Birds chirp loudly in the distance", "A person talks nearby"]

Output: """
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
        """Normalize output text (convert to list format)"""
        result_text = result_text.strip()
        # Remove newlines to make single line
        result_text = " ".join(result_text.split())

        # Extract all quoted strings using regex
        quoted_strings = re.findall(r'"([^"]*)"', result_text)
        if quoted_strings:
            return json.dumps(quoted_strings)

        # Fallback: try to parse as JSON array
        start_idx = result_text.find("[")
        end_idx = result_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            array_str = result_text[start_idx : end_idx + 1]
            try:
                items = json.loads(array_str)
                if isinstance(items, list):
                    return json.dumps(items)
            except json.JSONDecodeError:
                pass

        # Last resort: wrap entire text
        return json.dumps([result_text])

    def _fix_unquoted_strings(self, result_text: str) -> str:
        """Wrap unquoted strings with double quotes"""
        # Remove extra { } ( )
        for char in "{}()":
            result_text = result_text.replace(char, "")
        # Return as-is if already parseable
        try:
            json.loads(result_text)
            return result_text
        except json.JSONDecodeError:
            pass
        # Remove [ and ] to get inner content
        inner = result_text[1:-1].strip()
        if not inner:
            return "[]"
        # Split by comma and process each element
        items = []
        for item in inner.split(","):
            item = item.strip().strip('"').strip("'")
            # For "key": "value" format, get only the value part
            if ":" in item:
                item = item.split(":")[-1].strip().strip('"').strip("'")
            if item:
                items.append(item)
        quoted_items = [f'"{item}"' for item in items]
        return "[" + ", ".join(quoted_items) + "]"

    def _parse_json_result(self, result_text: str) -> list:
        """Parse JSON string and convert to list"""
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
        """Decode and parse a single response"""
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


# class SamAudio:
#     def __init__(
#         self, model_name: str = "facebook/sam-audio-small", dtype=torch.bfloat16
#     ):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.dtype = dtype
#         self.model_name = model_name
#         self.model = SAMAudio.from_pretrained(model_name)
#         self.processor = SAMAudioProcessor.from_pretrained(model_name)
#         self.model = self.model.eval().to(self.device, self.dtype)
#         self.sample_rate = self.processor.audio_sampling_rate

#     def separate_audio(
#         self,
#         audio_file: str,
#         descriptions: list[str],
#         predict_spans: bool = False,
#         reranking_candidates: int = 1,
#     ) -> dict[str, torch.Tensor]:
#         """
#         Separate audio based on text descriptions.
#         Args:
#             audio_file: Path to audio file.
#             descriptions: List of text descriptions for separation.
#             predict_spans: Whether to predict spans (better quality but slower).
#             reranking_candidates: Number of reranking candidates.
#         Returns:
#             Dict mapping description to separated audio tensor.
#         """
#         separated_audios = {}
#         for description in descriptions:
#             with torch.no_grad():
#                 batch = self.processor(
#                     audios=[audio_file],
#                     descriptions=[description],
#                 ).to(self.device)
#                 # Only convert audio tensors to dtype, keep index tensors as int64
#                 batch.audios = batch.audios.to(self.dtype)
#                 result = self.model.separate(
#                     batch,
#                     predict_spans=predict_spans,
#                     reranking_candidates=reranking_candidates,
#                 )

#             separated_audios[description] = result.target[0]
#         return separated_audios

#     def save_audio(
#         self,
#         save_path: str,
#         audio_tensor: torch.Tensor,
#         dtype: torch.dtype = torch.float32,
#     ):
#         """Save audio tensor to file"""
#         sample_rate = self.processor.audio_sampling_rate
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         # Ensure tensor is 2D (num_channels, num_samples)
#         if audio_tensor.dim() == 1:
#             audio_tensor = audio_tensor.unsqueeze(0)
#         torchaudio.save(save_path, audio_tensor.cpu().to(dtype), sample_rate)


### feature saving functions ###


def check_all_file_exists(
    datasets: list[str],
    audio_files: list[str],
    feats_dir: str,
    feats_name: str,
) -> bool:
    """Check if all feature files in the batch exist"""
    all_exist = True
    for i in range(len(audio_files)):
        dataset = datasets[i]
        file_name = os.path.basename(audio_files[i]).replace(".wav", ".pt")
        save_path = os.path.join(feats_dir, feats_name, dataset, file_name)
        if not os.path.exists(save_path):
            all_exist = False
            break
    return all_exist


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


def msclap_extract(dataloader, feats_dir: str, seed: int = 42):
    msclap_embedder = MSClapEmbedder(seed=seed)

    for batch in tqdm(dataloader, desc="Extracting MSCLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in audio_files
        ]
        text_file_names = [f"{text_id}.pt" for text_id in text_ids]

        if check_all_file_exists(
            datasets,
            audio_files,
            feats_dir,
            "msclap_audio",
        ) and check_all_file_exists(
            datasets,
            text_file_names,
            feats_dir,
            "msclap_text",
        ):
            continue

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

        if check_all_file_exists(
            datasets,
            audio_files,
            feats_dir,
            "laionclap_audio",
        ) and check_all_file_exists(
            datasets,
            text_file_names,
            feats_dir,
            "laionclap_text",
        ):
            continue

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


def openl3_extract(dataloader, feats_dir: str):
    openl3_embedder = Openl3Embedder()
    for batch in tqdm(dataloader, desc="Extracting Openl3 features"):
        datasets = batch["dataset"]
        content_type = (
            "env" if datasets[0] in ["relate", "audiocap", "xacle"] else "music"
        )
        sr = dataloader.dataset.bitrate

        audio_files = batch["audio_file_path"]
        audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in audio_files
        ]
        if not check_all_file_exists(datasets, audio_files, feats_dir, "openl3_audio"):
            openl3_audio_embeddings = openl3_embedder.embed_audios(
                audio_files,
                samplingrate=sr,
                content_type=content_type,
            )
            save_batch_feats(
                feats_dir,
                datasets,
                "openl3_audio",
                audio_file_names,
                openl3_audio_embeddings,
            )

        ref_audio_files = batch["ref_audio_file_path"]
        ref_audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in ref_audio_files
        ]
        if not check_all_file_exists(
            datasets, ref_audio_files, feats_dir, "openl3_audio_ref"
        ):
            ref_audio_embeddings = openl3_embedder.embed_audios(
                ref_audio_files,
                samplingrate=sr,
                content_type=content_type,
            )
            save_batch_feats(
                feats_dir,
                datasets,
                "openl3_audio_ref",
                ref_audio_file_names,
                ref_audio_embeddings,
            )


def embed_parsed_data(
    dataloader,
    feats_dir: str,
    seq_size: int = 20,
    embed_model: str = "laionclap",
):
    """
    Embed parsed audio segments and text prompts.
    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory containing features (separated_audio, parsed_texts).
        seq_size: Maximum sequence size for padding/truncating.
        embed_model: Embedding model to use ("laionclap" or "msclap").
    """
    if embed_model == "laionclap":
        embedder = LaionClapEmbedder()
        feats_prefix = "laionclap"
    elif embed_model == "msclap":
        embedder = MSClapEmbedder()
        feats_prefix = "msclap"

    for batch in tqdm(dataloader, desc="Embedding Parsed Audio Segments"):
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        for text_id, dataset in zip(text_ids, datasets):
            # Load parsed audio sources
            text_path = os.path.join(
                feats_dir, "parsed_texts", dataset, f"{text_id}.json"
            )
            with open(text_path, "r") as f:
                audio_sources: list[str] = json.load(f)

            # Load separated audio files
            audio_dir = os.path.join(feats_dir, "separated_audio", dataset, text_id)
            audio_files = sorted(
                [
                    os.path.join(audio_dir, f)
                    for f in os.listdir(audio_dir)
                    if f.endswith(".wav")
                ]
            )

            if len(audio_files) == 0 and len(audio_sources) == 0:
                print(text_id, dataset)
                continue
            if not len(audio_files) == len(audio_sources):
                raise ValueError(
                    f"Number of separated audio files ({len(audio_files)}) does not match number of audio sources ({len(audio_sources)}) for {text_id} in {dataset}."
                )

            # Embed audio and text
            audio_embeddings = embedder.embed_audios(audio_files)  # [N, D]
            text_embeddings = embedder.embed_texts(audio_sources)  # [N, D]

            num_segments = audio_embeddings.shape[0]
            embed_dim = audio_embeddings.shape[-1]

            # Create mask for valid positions
            mask = torch.zeros(seq_size, dtype=torch.bool)
            valid_len = min(num_segments, seq_size)
            mask[:valid_len] = True

            # Pad or truncate to seq_size
            if num_segments < seq_size:
                pad_size = seq_size - num_segments
                audio_pad = torch.zeros(
                    pad_size,
                    embed_dim,
                    device=audio_embeddings.device,
                    dtype=audio_embeddings.dtype,
                )
                text_pad = torch.zeros(
                    pad_size,
                    embed_dim,
                    device=text_embeddings.device,
                    dtype=text_embeddings.dtype,
                )
                audio_embeddings = torch.cat([audio_embeddings, audio_pad], dim=0)
                text_embeddings = torch.cat([text_embeddings, text_pad], dim=0)
            else:
                audio_embeddings = audio_embeddings[:seq_size]
                text_embeddings = text_embeddings[:seq_size]

            # Save embeddings and mask
            save_feats(
                feats_dir=feats_dir,
                feats_name=f"{feats_prefix}_parsed_audio",
                dataset=dataset,
                file_name=f"{text_id}.pt",
                feats=audio_embeddings,
            )
            save_feats(
                feats_dir=feats_dir,
                feats_name=f"{feats_prefix}_parsed_text",
                dataset=dataset,
                file_name=f"{text_id}.pt",
                feats=text_embeddings,
            )
            save_feats(
                feats_dir=feats_dir,
                feats_name="parsed_mask",
                dataset=dataset,
                file_name=f"{text_id}.pt",
                feats=mask,
            )


def clear_gpu_memory():
    """Clear GPU memory cache"""
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    dataset = TTAPreprocessDataset(data_dir=args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    openl3_extract(dataloader, args.feats_dir)

    # msclap_extract(dataloader, args.feats_dir)
    # clear_gpu_memory()
    # text_parse(dataloader, args.feats_dir)
    # clear_gpu_memory()
    # embed_parsed_data(dataloader, args.feats_dir, embed_model="msclap")
    # clear_gpu_memory()
    # embed_parsed_data(dataloader, args.feats_dir, embed_model="laionclap")
    # create_diff_audio(dataloader, args.feats_dir)


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
        default=1,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    fix_seed(args.seed)
    splits = ["test"]
    for split in splits:
        args.split = split
        main(args)
