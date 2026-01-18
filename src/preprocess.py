import argparse
import json
import os
import re
import time
from typing import List

import laion_clap
import torch
import torchaudio
from google import genai
from msclap import CLAP
from openai import OpenAI
from pydantic import BaseModel, Field
from sam_audio import SAMAudio, SAMAudioProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, ClapModel, ClapProcessor

from dataset import TTADataset
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


class HumanCLAPEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        model_path = "sarulab-speech/human-clap-wsce-mae"
        processor_path = "laion/clap-htsat-fused"
        self.model = ClapModel.from_pretrained(model_path).to(0)
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(processor_path)
        self.dtype = dtype
        self.target_sr = 48000
        self.resampler_16k = torchaudio.transforms.Resample(16000, self.target_sr)

    def _load_audio(self, audio_path: str) -> list:
        """Load audio file and resample to 48kHz if needed"""
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]  # mono
        if sr == 16000:
            audio = self.resampler_16k(audio)
        elif sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        return audio.detach().numpy().copy()

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch (truncate long texts)"""
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(0)
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """Embed audios in batch"""
        audios = [self._load_audio(f) for f in audio_files]
        with torch.no_grad():
            inputs = self.processor(
                audios=audios,
                return_tensors="pt",
                sampling_rate=self.target_sr,
                padding=True,
            ).to(0)
            audio_embeddings = self.model.get_audio_features(**inputs)
        return audio_embeddings.to(self.dtype)


class SoundEvents(BaseModel):
    """Pydantic model for structured output of sound events."""

    events: List[str] = Field(
        description="List of sound events extracted from the caption."
    )


class GeminiTextParser:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            api_key = input("Enter your Google API key: ")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = """Split the caption into separate sound events. Keep all modifiers (adjectives, adverbs, descriptions).

Example:
Caption: A large dog barks loudly while heavy rain falls on the metal roof.
Output: ["A large dog barks loudly", "heavy rain falls on the metal roof"]"""

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        """Parse texts in batch and return list of sound events for each text."""
        responses = []
        for text in texts:
            prompt = f"{self.system_prompt}\n\nCaption: {text}\n\nOutput:"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": SoundEvents.model_json_schema(),
                },
            )
            sound_events = SoundEvents.model_validate_json(response.text)
            responses.append(sound_events.events)
        return responses


class GPTTextParser:
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        self.client = OpenAI()
        self.model_name = model_name
        self.system_prompt = """Identify all sound events in the caption.

Rules:
- Each element = ONE sound event (concise NP or VP form)
- No duplicates or semantically overlapping events
- No emotional/evaluative/subjective modifiers

Example:
Caption: Birds chirp loudly in the distance; a person talks nearby; more chirping.
Output: ["Birds chirping loudly in the distance", "A person talking nearby"]"""

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        responses = []
        total = len(texts)
        for idx, text in enumerate(texts, start=1):
            print(f"[gpt-text-parse] {idx}/{total} start")
            start_time = time.perf_counter()
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Caption: {text}"},
                ],
                response_format=SoundEvents,
            )
            result = response.choices[0].message.parsed.events
            elapsed = time.perf_counter() - start_time
            print(f"[gpt-text-parse] {idx}/{total} done ({elapsed:.2f}s)")
            responses.append(result)
        return responses


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


class SamAudio:
    def __init__(
        self, model_name: str = "facebook/sam-audio-large", dtype=torch.bfloat16
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.model_name = model_name
        self.model = SAMAudio.from_pretrained(model_name)
        self.processor = SAMAudioProcessor.from_pretrained(model_name)
        self.model = self.model.eval().to(self.device, self.dtype)
        self.sample_rate = self.processor.audio_sampling_rate

    def separate_audio(
        self,
        audio_file: str,
        prompts: list[str],
        predict_spans: bool = True,
        reranking_candidates: int = 1,
    ) -> list[torch.Tensor]:
        """
        Separate audio based on text prompts.
        Args:
            audio_file: Path to audio file.
            prompts: List of text prompts for separation.
            predict_spans: Whether to predict spans (better quality but slower).
            reranking_candidates: Number of reranking candidates.
        Returns:
            List of separated audio tensors.
        """
        separated_audios = []
        for prompt in prompts:
            with torch.no_grad():
                batch = self.processor(
                    audios=[audio_file],
                    descriptions=[prompt],
                ).to(self.device)
                # Only convert audio tensors to dtype, keep index tensors as int64
                batch.audios = batch.audios.to(self.dtype)
                result = self.model.separate(
                    batch,
                    predict_spans=predict_spans,
                    reranking_candidates=reranking_candidates,
                )

            separated_audios.append(result.target[0])
        return separated_audios

    def save_audio(
        self,
        save_path: str,
        audio_tensor: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        """Save audio tensor to file"""
        sample_rate = self.processor.audio_sampling_rate
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Ensure tensor is 2D (num_channels, num_samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        torchaudio.save(save_path, audio_tensor.cpu().to(dtype), sample_rate)


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


def humanclap_extract(dataloader, feats_dir: str):
    human_clap_embedder = HumanCLAPEmbedder()
    for batch in tqdm(dataloader, desc="Extracting HumanCLAP features"):
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
            "humanclap_audio",
        ) and check_all_file_exists(
            datasets,
            text_file_names,
            feats_dir,
            "humanclap_text",
        ):
            continue

        human_audio_embeddings = human_clap_embedder.embed_audios(audio_files)
        human_text_embeddings = human_clap_embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            "humanclap_audio",
            audio_file_names,
            human_audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            "humanclap_text",
            text_file_names,
            human_text_embeddings,
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


def text_parse(dataloader, feats_dir: str):
    text_parser = GeminiTextParser()
    # text_parser = QwenTextParser()
    for batch in tqdm(dataloader, desc="Parsing Text with Gemini"):
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        # Check if all files in batch already exist
        all_exist = True
        for text_id, dataset in zip(text_ids, datasets):
            save_path = os.path.join(
                feats_dir, "parsed_texts", dataset, f"{text_id}.json"
            )
            if not os.path.exists(save_path):
                all_exist = False
                break
        if all_exist:
            continue

        # Parse texts using Gemini
        audio_sources: list[list[str]] = text_parser.parse_texts(texts)

        for text_id, dataset, sources in zip(text_ids, datasets, audio_sources):
            save_path = os.path.join(
                feats_dir, "parsed_texts", dataset, f"{text_id}.json"
            )
            sources = list(set(sources))  # remove duplicates
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(sources, f, ensure_ascii=False, indent=0)


def audio_parse(dataloader, feats_dir: str):
    """Parse audio using SAM-Audio based on parsed text sources"""
    sam_audio = SamAudio()
    for batch in tqdm(dataloader, desc="Parsing Audio with SAM-Audio"):
        audio_files = batch["audio_file_path"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        for text_id, dataset, audio_file in zip(text_ids, datasets, audio_files):
            # Load parsed audio sources
            text_path = os.path.join(
                feats_dir, "parsed_texts", dataset, f"{text_id}.json"
            )
            with open(text_path, "r") as f:
                audio_sources = json.load(f)
            audio_sources = [src.lower() for src in audio_sources]
            save_dir = os.path.join(feats_dir, "separated_audio", dataset, text_id)
            os.makedirs(save_dir, exist_ok=True)

            if len(os.listdir(save_dir)) == len(audio_sources):
                continue

            # Split audio using SAM-Audio
            separated_audios: list[torch.Tensor] = sam_audio.separate_audio(
                audio_file, audio_sources
            )

            # Save separated audio files
            for i, audio_tensor in enumerate(separated_audios):
                save_path = os.path.join(save_dir, f"{i}.wav")
                sam_audio.save_audio(save_path, audio_tensor)


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

            # Embed audio and text
            audio_embeddings = embedder.embed_audios(audio_files)  # [N, D]
            text_embeddings = embedder.embed_texts(audio_sources)  # [N, D]

            audio_len = audio_embeddings.shape[0]
            text_len = text_embeddings.shape[0]
            num_segments = min(audio_len, text_len)
            embed_dim = audio_embeddings.shape[-1]

            if audio_len != text_len:
                audio_embeddings = audio_embeddings[:num_segments]
                text_embeddings = text_embeddings[:num_segments]

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
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

    humanclap_extract(dataloader, args.feats_dir)
    clear_gpu_memory()
    laionclap_extract(dataloader, args.feats_dir)
    clear_gpu_memory()
    msclap_extract(dataloader, args.feats_dir, seed=args.seed)
    clear_gpu_memory()
    text_parse(dataloader, args.feats_dir)
    clear_gpu_memory()
    audio_parse(dataloader, args.feats_dir)
    clear_gpu_memory()
    embed_parsed_data(dataloader, args.feats_dir, embed_model="msclap")
    clear_gpu_memory()
    embed_parsed_data(dataloader, args.feats_dir, embed_model="laionclap")


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
    splits = ["train", "val", "test"]
    for split in splits:
        args.split = split
        main(args)
