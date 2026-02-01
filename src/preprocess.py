import argparse
import json
import os
import re
import time
from abc import ABC
from typing import List

import laion_clap
import librosa
import torch
import torchaudio
from google import genai
from google.genai.types import GenerateContentConfig
from msclap import CLAP
from openai import OpenAI
from pydantic import BaseModel, Field
from sam_audio import SAMAudio, SAMAudioProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, ClapModel, ClapProcessor

from AudioSep.models.audiosep import AudioSep as AudioSepModel
from AudioSep.utils import get_ss_model
from CLAPSep.model.CLAPSep import CLAPSep
from dataset import TTADataset
from SoloAudio.solo_audio_ins import SoloAudio
from utils.helper_func import fix_seed

### laion clap fix ###
_torch_load = torch.load


def torch_load_no_wo(*args, **kwargs):
    """Wrapper for torch.load that disables weights_only parameter.

    This is a workaround for LAION-CLAP compatibility issues.

    Args:
        *args: Positional arguments passed to torch.load.
        **kwargs: Keyword arguments passed to torch.load.

    Returns:
        The loaded object from torch.load.
    """
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = torch_load_no_wo


class TTAPreprocessDataset(TTADataset):
    """Dataset class for TTA preprocessing.

    Extends TTADataset to provide raw data items for preprocessing.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        """Initialize TTAPreprocessDataset.

        Args:
            data_dir: Directory containing the dataset.
            split: Dataset split (train/val/test).
        """
        super().__init__(data_dir=data_dir, split=split)

    def __getitem__(self, idx: int) -> dict:
        """Get a data item by index.

        Args:
            idx: Index of the item.

        Returns:
            Dictionary containing the data item.
        """
        data = self.database[idx]
        return data


##### Abstract base classes ######


class CLAPEmbedder(ABC):
    """Base class for CLAP embedders.

    Subclasses must implement embed_texts and embed_audios methods.
    """

    name: str

    def __init__(self, name: str):
        """Initialize CLAPEmbedder.

        Args:
            name: Name of the embedder.
        """
        self.name = name

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch.

        Args:
            texts: List of text strings to embed.

        Returns:
            Text embeddings tensor of shape [B, D].
        """
        raise NotImplementedError

    def embed_audios(self, audio_files: list[str] | list[torch.Tensor]) -> torch.Tensor:
        """Embed audios in batch.

        Args:
            audio_files: List of audio file paths.

        Returns:
            Audio embeddings tensor of shape [B, D].
        """
        raise NotImplementedError


class AudioSeparator(ABC):
    """Base class for audio separators.

    Subclasses must implement separate_audio method.
    """

    name: str
    sample_rate: int

    def __init__(self, name: str, sample_rate: int):
        """Initialize AudioSeparator.

        Args:
            name: Name of the audio separator.
            sample_rate: Sample rate for audio processing.
        """
        self.name = name
        self.sample_rate = sample_rate

    def separate_audio(
        self,
        audio_file: str,
        prompts: list[str],
    ) -> list[torch.Tensor]:
        """Separate audio based on text prompts.

        Args:
            audio_file: Path to audio file.
            prompts: List of text prompts for separation.

        Returns:
            List of separated audio tensors.
        """
        raise NotImplementedError

    def save_audio(
        self,
        save_path: str,
        audio_tensor: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        """Save audio tensor to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Ensure tensor is 2D (num_channels, num_samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        torchaudio.save(save_path, audio_tensor.cpu().to(dtype), self.sample_rate)


class TextParser(ABC):
    """Base class for text parsers.

    Subclasses must implement parse_texts method to extract sound events.
    """

    name: str

    def __init__(self, name: str):
        """Initialize TextParser.

        Args:
            name: Name of the text parser.
        """
        self.name = name

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        """Parse texts in batch and return list of sound events for each text.

        Args:
            texts: List of caption texts to parse.

        Returns:
            List of sound event lists for each input text.
        """
        raise NotImplementedError


####### Implementations of CLAP Embedders ######


class MSClapEmbedder(CLAPEmbedder):
    """CLAP embedder using Microsoft CLAP model."""

    def __init__(self, dtype: torch.dtype = torch.float32, seed: int = 42):
        """Initialize MSClapEmbedder.

        Args:
            dtype: Data type for embeddings.
            seed: Random seed for reproducibility.
        """
        super().__init__(name="msclap")
        self.model = CLAP(version="2023", use_cuda=True)
        self.max_text_len = 77  # MSCLAP max text length
        self.dtype = dtype
        self.seed = seed

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch.

        Args:
            texts: List of text strings to embed.

        Returns:
            Text embeddings tensor of shape [B, D].
        """
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embeddings(truncated_texts)
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str] | list[torch.Tensor]) -> torch.Tensor:
        """Embed audios in batch.

        Args:
            audio_files: List of audio file paths or tensor.

        Returns:
            Audio embeddings tensor of shape [B, D].
        """
        fix_seed(self.seed)  # Ensure seed is fixed before audio embedding
        with torch.no_grad():
            if isinstance(audio_files[0], torch.Tensor):
                raise NotImplementedError(
                    "MS-CLAP embedder does not support audio tensors."
                )
            audio_embeddings = self.model.get_audio_embeddings(audio_files)
        return audio_embeddings.to(self.dtype)


class LaionClapEmbedder(CLAPEmbedder):
    """CLAP embedder using LAION CLAP model."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initialize LaionClapEmbedder.

        Args:
            dtype: Data type for embeddings.
        """
        super().__init__(name="laionclap")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt("models/630k-audioset-best.pt")
        self.model.eval()
        self.max_text_len = 77  # LaionCLAP max text length
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch with truncation for long texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Text embeddings tensor of shape [B, D].
        """
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(
                truncated_texts, use_tensor=True
            )
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str] | list[torch.Tensor]) -> torch.Tensor:
        """Embed audios in batch.

        Args:
            audio_files: List of audio file paths or tensor.

        Returns:
            Audio embeddings tensor of shape [B, D].
        """
        with torch.no_grad():
            if isinstance(audio_files[0], torch.Tensor):
                raise NotImplementedError(
                    "LAION-CLAP embedder does not support audio tensors."
                )
            audio_embeddings = self.model.get_audio_embedding_from_filelist(
                x=audio_files, use_tensor=True
            )
        return audio_embeddings.to(self.dtype)


class HumanCLAPEmbedder(CLAPEmbedder):
    """CLAP embedder using Human-CLAP model."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initialize HumanCLAPEmbedder.

        Args:
            dtype: Data type for embeddings.
        """
        super().__init__(name="humanclap")
        model_path = "sarulab-speech/human-clap-wsce-mae"
        processor_path = "laion/clap-htsat-fused"
        self.model = ClapModel.from_pretrained(model_path).to(0)
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(processor_path)
        self.dtype = dtype
        self.target_sr = 48000
        self.resampler_16k = torchaudio.transforms.Resample(16000, self.target_sr)

    def _load_audio(self, audio_path: str) -> list:
        """Load audio file and resample to 48kHz if needed.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Audio waveform as numpy array.
        """
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]  # mono
        if sr == 16000:
            audio = self.resampler_16k(audio)
        elif sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        return audio.detach().numpy().copy()

    def resample_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio tensor to target sample rate.

        Args:
            audio: Audio tensor.

        Returns:
            Resampled audio tensor.
        """
        if orig_sr == self.target_sr:
            return audio
        resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return resampler(audio)

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed texts in batch with truncation for long texts.

        Args:
            texts: List of text strings or tensor to embed.

        Returns:
            Text embeddings tensor of shape [B, D].
        """
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

    def embed_audios(self, audio_files: list[str] | list[torch.Tensor]) -> torch.Tensor:
        """Embed audios in batch.

        Args:
            audio_files: List of audio file paths.

        Returns:
            Audio embeddings tensor of shape [B, D].
        """
        if isinstance(audio_files[0], torch.Tensor):
            audios = [
                audio_file.detach().cpu().float().numpy().copy()
                for audio_file in audio_files
            ]
        else:
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


class GeminiTextParser(TextParser):
    """Text parser using Google Gemini model."""

    def __init__(
        self, model_name: str = "gemini-2.5-flash", cooldown_time: float = 0.1
    ):
        """Initialize GeminiTextParser.

        Args:
            model_name: Name of the Gemini model to use.
        """
        super().__init__(name="gemini")
        self.cooldown_time = cooldown_time
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key is None:
            api_key = input("Enter your Google API key: ")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = (
            "You are a text parser. Output ONLY a JSON array of strings."
        )

    def build_prompt(self, text: str) -> str:
        """Build prompt from caption text.

        Args:
            text: Caption text to include in the prompt.

        Returns:
            Formatted prompt string for GPT.
        """
        prompt = f"""Task:
        Identify all sound events described in the following caption.

        Rules:
        - Each element must correspond to ONE sound event.
        - Express each sound event in a concise NP or VP form.
        - Do NOT include duplicate or semantically overlapping sound events.
        - Do NOT include emotional, evaluative, or subjective modifiers.
        - If the caption describes only ONE sound event, output a JSON array with a single string.
        - Output MUST be a valid JSON array of strings.

        Example 1:
        Caption: Birds chirp loudly in the distance; a person talks nearby; more chirping.
        Output: ["Birds chirping loudly in the distance", "A person talking nearby"]

        Example 2:
        A male vocalist sings this spirited song. The song is medium tempo with energetic electric guitar lead enthusiastic electric bass guitar  hard hitting drums and keyboard harmony. The vocals are passionate youthfulenergetic vociferous powerful and loud . This song is Hard Rock/Metal.
        Output: ["A male vocalist singing", "An electric guitar lead playing", "An electric bass guitar playing", "Drums playing", "A keyboard harmony playing"]

        Caption: {text}

        Output: """
        return prompt

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        """Parse texts in batch using Gemini model.

        Args:
            texts: List of caption texts to parse.

        Returns:
            List of sound event lists for each input text.
        """
        responses = []
        for text in texts:
            prompt = self.build_prompt(text)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=SoundEvents.model_json_schema(),
                    temperature=0.0,
                ),
            )
            sound_events = SoundEvents.model_validate_json(response.text)
            responses.append(list(set(sound_events.events)))
            time.sleep(self.cooldown_time)
        return responses


class GPTTextParser(TextParser):
    """Text parser using OpenAI GPT model."""

    def __init__(self, model_name: str = "gpt-5.2"):
        """Initialize GPTTextParser.

        Args:
            model_name: Name of the GPT model to use.
        """
        super().__init__(name="gpt")
        self.client = OpenAI()
        self.model_name = model_name
        self.system_prompt = (
            "You are a text parser. Output ONLY a JSON array of strings."
        )

    def build_prompt(self, text: str) -> str:
        """Build prompt from caption text.

        Args:
            text: Caption text to include in the prompt.

        Returns:
            Formatted prompt string for GPT.
        """
        prompt = f"""Task:
        Identify all sound events described in the following caption.

        Rules:
        - Each element must correspond to ONE sound event.
        - Express each sound event in a concise NP or VP form.
        - Do NOT include duplicate or semantically overlapping sound events.
        - Do NOT include emotional, evaluative, or subjective modifiers.
        - If the caption describes only ONE sound event, output a JSON array with a single string.
        - Output MUST be a valid JSON array of strings.

        Example 1:
        Caption: Birds chirp loudly in the distance; a person talks nearby; more chirping.
        Output: ["Birds chirping loudly in the distance", "A person talking nearby"]

        Example 2:
        A male vocalist sings this spirited song. The song is medium tempo with energetic electric guitar lead enthusiastic electric bass guitar  hard hitting drums and keyboard harmony. The vocals are passionate youthfulenergetic vociferous powerful and loud . This song is Hard Rock/Metal.
        Output: ["A male vocalist singing", "An electric guitar lead playing", "An electric bass guitar playing", "Drums playing", "A keyboard harmony playing"]

        Caption: {text}

        Output: """
        return prompt

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        """Parse texts in batch using GPT model.

        Args:
            texts: List of caption texts to parse.

        Returns:
            List of sound event lists for each input text.
        """
        responses = []
        for text in texts:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.build_prompt(text)},
                ],
                response_format=SoundEvents,
                temperature=0.0,
            )
            result = response.choices[0].message.parsed.events
            responses.append(list(set(result)))
        return responses


class QwenTextParser(TextParser):
    """Text parser using Qwen model for local inference."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        """Initialize QwenTextParser.

        Args:
            model_name: Name or path of the Qwen model to use.
        """
        super().__init__(name="qwen")
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
        """Build chat template from input text.

        Args:
            text: Caption text to process.

        Returns:
            Formatted chat template string for the model.
        """
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
        """Normalize output text to JSON list format.

        Args:
            result_text: Raw output text from the model.

        Returns:
            JSON string representing a list of sound events.
        """
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
        """Wrap unquoted strings with double quotes.

        Args:
            result_text: Text containing potentially unquoted strings.

        Returns:
            JSON string with properly quoted elements.
        """
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
        """Parse JSON string and convert to list.

        Args:
            result_text: JSON string to parse.

        Returns:
            List of sound events.

        Raises:
            RuntimeError: If JSON parsing fails.
        """
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
        """Decode and parse a single response from generated token IDs.

        Args:
            ids: Generated token IDs including input.
            input_len: Length of the input tokens to skip.

        Returns:
            List of parsed sound events.
        """
        result_text = self.tokenizer.decode(
            ids[input_len:],
            skip_special_tokens=True,
        )
        result_text = self._normalize_output(result_text)
        return self._parse_json_result(result_text)

    def parse_texts(self, texts: list[str]) -> list[list[str]]:
        """Parse texts in batch using Qwen model.

        Args:
            texts: List of caption texts to parse.

        Returns:
            List of sound event lists for each input text.
        """
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
            responses.append(list(set(result)))

        return responses


class SAMAudioSeparator(AudioSeparator):
    """Audio separation using SAM-Audio model."""

    def __init__(
        self, model_name: str = "facebook/sam-audio-large-tv", dtype=torch.bfloat16
    ):
        """Initialize SamAudio.

        Args:
            model_name: Name or path of the SAM-Audio model.
            dtype: Data type for model inference.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.model_name = model_name
        self.model = SAMAudio.from_pretrained(model_name)
        self.processor = SAMAudioProcessor.from_pretrained(model_name)
        self.model = self.model.eval().to(self.device, self.dtype)
        super().__init__(
            name="sam_audio", sample_rate=self.processor.audio_sampling_rate
        )

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
            prompt = prompt.lower()
            with torch.inference_mode():
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


class CLAPSepSeparator(AudioSeparator):
    def __init__(
        self,
        model_path: str = "models/clapsep-best.ckpt",
        clap_model_path: str = "models/clapsep-clap.pt",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize CLAPSepSeparator.

        Args:
            model_path: Path to the CLAPSep model checkpoint.
            dtype: Data type for model inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        model_config = {
            "lan_embed_dim": 1024,
            "depths": [1, 1, 1, 1],
            "embed_dim": 128,
            "encoder_embed_dim": 128,
            "phase": False,
            "spec_factor": 8,
            "d_attn": 640,
            "n_masker_layer": 3,
            "conv": False,
        }

        self.model = CLAPSep(model_config, clap_model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.model.to(self.device, self.dtype)
        super().__init__(name="clapsep", sample_rate=32000)

    def separate_audio(
        self,
        audio_file: str,
        prompts: list[str],
    ) -> list[torch.Tensor]:
        """
        Separate audio based on text prompts.
        Args:
            audio_file: Path to audio file.
            prompts: List of text prompts for separation.
        Returns:
            List of separated audio tensors.
        """
        separated_audios = []
        for prompt in prompts:
            with torch.inference_mode():
                audio, fs = librosa.load(audio_file, sr=32000)
                audio_tensor = torch.tensor(
                    audio, device=self.device, dtype=self.dtype
                ).unsqueeze(0)
                separated = self.model.inference_from_data(
                    audio_tensor,
                    pos_prompt=[prompt],
                    neg_prompt=[],
                )
                separated_audios.append(separated[0].cpu())
        return separated_audios


class SoloAudioSeparator(AudioSeparator):
    def __init__(
        self,
        clap_model_path: str = "laion/larger_clap_general",
        autoencoder_path: str = "models/soloaudio_vae.pt",
        soloaudio_path: str = "models/soloaudio.pt",
        sample_rate: int = 24000,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize SoloAudioSeparator.

        Args:
            clap_model_path: Path to the CLAP model checkpoint.
            autoencoder_path: Path to the autoencoder model checkpoint.
            soloaudio_path: Path to the soloaudio model checkpoint.
            dtype: Data type for model inference.
        """
        super().__init__(name="soloaudio", sample_rate=sample_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.model = SoloAudio(clap_model_path, autoencoder_path, soloaudio_path)
        self.model.eval()
        self.model.to(self.device, self.dtype)

    def separate_audio(
        self,
        audio_file: str,
        prompts: list[str],
    ) -> list[torch.Tensor]:
        """
        Separate audio based on text prompts.
        Args:
            audio_file: Path to audio file.
            prompts: List of text prompts for separation.
        Returns:
            List of separated audio tensors.
        """
        separated_audios = []
        for prompt in prompts:
            with torch.no_grad():
                audio, _ = librosa.load(audio_file, sr=self.sample_rate)
                audio_tensor = torch.tensor(
                    audio, device=self.device, dtype=self.dtype
                ).unsqueeze(0)
                separated_audio = self.model(audio_tensor, prompt).squeeze(0)
                separated_audios.append(separated_audio.cpu())
        return separated_audios


class AudioSepSeparator(AudioSeparator):
    def __init__(
        self,
        model_config_path: str = "src/AudioSep/config/audiosep_base.yaml",
        model_name: str = "nielsr/audiosep-demo",
        sample_rate: int = 32000,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize AudioSepSeparator.

        Args:
            model_config_path: Path to the model configuration file.
            model_name: Name of the model to load.
            dtype: Data type for model inference.
        """
        super().__init__(name="audiosep", sample_rate=sample_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        ss_model = get_ss_model(model_config_path)
        self.model = AudioSepModel.from_pretrained(model_name, ss_model=ss_model)
        self.model.eval()
        self.model.to(self.device, self.dtype)

    def separate_audio(
        self,
        audio_file: str,
        prompts: list[str],
    ) -> list[torch.Tensor]:
        """
        Separate audio based on text prompts.
        Args:
            audio_file: Path to audio file.
            prompts: List of text prompts for separation.
        Returns:
            List of separated audio tensors.
        """
        separated_audios = []
        for prompt in prompts:
            with torch.no_grad():
                audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                conditions = self.model.query_encoder.get_query_embed(
                    modality="text", text=[prompt], device=self.device
                )
                input_dict = {
                    "mixture": torch.Tensor(audio)[None, None, :].to(self.device),
                    "condition": conditions,
                }
                # separated_audio = self.model.ss_model.chunk_inference(input_dict)
                # separated_audio = torch.tensor(separated_audio.squeeze(0))
                sep_segment = self.model.ss_model(input_dict)["waveform"]
                separated_audio = sep_segment.squeeze(0).squeeze(0).data.cpu()
                # separated_audio = torch.round(separated_audio * 32767)
                separated_audios.append(separated_audio.cpu())
        return separated_audios


####### Feature saving/loading utilities ######


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
    """Save feature tensor to disk.

    Args:
        feats_dir: Root directory for features.
        feats_name: Name of the feature type (e.g., "msclap_audio").
        dataset: Dataset name.
        file_name: Output file name.
        feats: Feature tensor to save.

    Returns:
        Path to the saved file.
    """
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


def clear_gpu_memory():
    """Clear GPU memory cache.

    Runs garbage collection and empties CUDA cache to free up GPU memory.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()


####### Main extraction and parsing functions ######


def clap_extract(dataloader, feats_dir: str, embedder: CLAPEmbedder):
    """
    Extract CLAP features for audio and text in the dataloader.
    For CompA dataset, also extract features for all choice options.
    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory to save features.
        embedder: CLAP embedder instance.
    """
    for batch in tqdm(dataloader, desc="Extracting CLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        audio_file_names = [
            os.path.basename(path).replace(".wav", ".pt") for path in audio_files
        ]
        text_file_names = [f"{text_id}.pt" for text_id in text_ids]

        if check_all_file_exists(
            datasets, audio_files, feats_dir, f"{embedder.name}_audio"
        ) and check_all_file_exists(
            datasets, text_file_names, feats_dir, f"{embedder.name}_text"
        ):
            continue

        # Extract main features
        audio_embeddings = embedder.embed_audios(audio_files)
        text_embeddings = embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            f"{embedder.name}_audio",
            audio_file_names,
            audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            f"{embedder.name}_text",
            text_file_names,
            text_embeddings,
        )

        rev_texts = batch["rev_text"]
        rev_audios = batch["rev_audio"]
        for i, dataset in enumerate(datasets):
            if dataset != "compa":
                continue
            assert rev_texts[i] != "" and rev_audios[i] != ""
            rev_text_embeddings = embedder.embed_texts([rev_texts[i]])
            rev_audio_embeddings = embedder.embed_audios([rev_audios[i]])
            rev_text_file_name = f"{text_ids[i]}_rev.pt"
            rev_audio_file_name = os.path.basename(rev_audios[i]).replace(".wav", ".pt")
            save_feats(
                feats_dir=feats_dir,
                feats_name=f"{embedder.name}_text",
                dataset=dataset,
                file_name=rev_text_file_name,
                feats=rev_text_embeddings,
            )
            save_feats(
                feats_dir=feats_dir,
                feats_name=f"{embedder.name}_audio",
                dataset=dataset,
                file_name=rev_audio_file_name,
                feats=rev_audio_embeddings,
            )


def text_parse(dataloader, feats_dir: str, text_parser: TextParser):
    """
    Parse text using specified model to extract sound events.
    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory to save features.
        text_parser: TextParser instance.
    """
    cache: dict[str, list[str]] = {}  # text -> parsed events

    for batch in tqdm(dataloader, desc="Parsing Text"):
        texts = batch["text"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]
        rev_texts = batch["rev_text"]

        # Collect texts that need parsing (not in cache and file doesn't exist)
        to_parse = []
        for text, text_id, dataset, rev_text in zip(
            texts, text_ids, datasets, rev_texts
        ):
            save_path = os.path.join(
                feats_dir,
                f"{text_parser.name}_parsed_texts",
                dataset,
                f"{text_id}.json",
            )
            if text not in to_parse:
                to_parse.append(text)
            if rev_text != "" and rev_text not in to_parse:
                to_parse.append(rev_text)

        # Parse unique texts
        if to_parse:
            results = text_parser.parse_texts(to_parse)
            for text, result in zip(to_parse, results):
                cache[text] = result

        # Save all texts in batch
        for text, text_id, dataset, rev_text in zip(
            texts, text_ids, datasets, rev_texts
        ):
            for text_str, suffix in [
                (text, ""),
                (rev_text, "_rev"),
            ]:
                if text_str == "":
                    continue
                save_path = os.path.join(
                    feats_dir,
                    f"{text_parser.name}_parsed_texts",
                    dataset,
                    f"{text_id + suffix}.json",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(cache[text_str], f, ensure_ascii=False, indent=0)


def audio_separate(
    dataloader,
    feats_dir: str,
    audio_separator: AudioSeparator,
    text_parser_model: str = "gpt",
):
    """Separate audio into individual sound sources using SAM-Audio.

    Uses parsed text sources as prompts to separate the original audio
    into individual sound components.

    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory containing parsed texts and for saving separated audio.
        text_parser_model: Name of the text parser used for parsing.
        audio_separator: AudioSeparator instance.
    """
    max_chunk_sec = 20
    for batch in tqdm(dataloader, desc="Parsing Audio with Audio Separator"):
        audio_files = batch["audio_file_path"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]
        texts = batch["text"]
        rev_texts = batch["rev_text"]
        rev_audios = batch["rev_audio"]

        for (
            text_id,
            dataset,
            audio_file,
            text,
            rev_text,
            rev_audio,
        ) in zip(
            text_ids,
            datasets,
            audio_files,
            texts,
            rev_texts,
            rev_audios,
        ):
            for text_str, suffix_text in [
                (text, ""),
                (rev_text, "_rev"),
            ]:
                if text_str == "":
                    continue
                # Load parsed audio sources
                text_path = os.path.join(
                    feats_dir,
                    f"{text_parser_model}_parsed_texts",
                    dataset,
                    f"{text_id + suffix_text}.json",
                )
                with open(text_path, "r", encoding="utf-8") as f:
                    audio_sources = json.load(f)
                save_dir = os.path.join(
                    feats_dir,
                    f"{audio_separator.name}_separated_audio",
                    dataset,
                    text_id + suffix_text,
                )
                os.makedirs(save_dir, exist_ok=True)

                existing_wavs = [f for f in os.listdir(save_dir) if f.endswith(".wav")]
                audio_source_count = len(audio_sources)
                audio_source_count += len(audio_sources) if rev_audio != "" else 0
                if len(existing_wavs) == audio_source_count:
                    continue
                if len(audio_sources) == 0:
                    continue

                for audio_file_path, prefix in [
                    (audio_file, ""),
                    (rev_audio, "rev_"),
                ]:
                    if audio_file_path == "":
                        continue

                    audio_info = torchaudio.info(audio_file_path)
                    duration_sec = audio_info.num_frames / audio_info.sample_rate

                    if duration_sec <= max_chunk_sec:
                        # Split audio using SAM-Audio with automatic mixed precision
                        with torch.amp.autocast("cuda"):
                            separated_audios: list[torch.Tensor] = (
                                audio_separator.separate_audio(
                                    audio_file_path, audio_sources
                                )
                            )

                        # Save separated audio files
                        for i, audio_tensor in enumerate(separated_audios):
                            save_path = os.path.join(save_dir, f"{prefix}{i}.wav")
                            audio_separator.save_audio(save_path, audio_tensor)
                        continue

                    # Long audio: chunk -> SAM -> stitch per prompt
                    audio, sr = torchaudio.load(audio_file_path)
                    target_sr = audio_separator.sample_rate
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        audio = resampler(audio)
                        sr = target_sr

                    samples_per_chunk = int(sr * max_chunk_sec)
                    total_samples = audio.shape[-1]
                    chunk_dir = os.path.join(save_dir, "chunks")
                    os.makedirs(chunk_dir, exist_ok=True)

                    chunk_paths = []
                    for chunk_idx, start in enumerate(
                        range(0, total_samples, samples_per_chunk)
                    ):
                        end = min(start + samples_per_chunk, total_samples)
                        if end <= start:
                            continue
                        chunk_audio = audio[..., start:end]
                        chunk_path = os.path.join(chunk_dir, f"{prefix}{chunk_idx}.wav")
                        torchaudio.save(chunk_path, chunk_audio, sr)
                        chunk_paths.append(chunk_path)

                    stitched: list[list[torch.Tensor]] = [[] for _ in audio_sources]
                    for chunk_path in chunk_paths:
                        with torch.amp.autocast("cuda"):
                            chunk_separated = audio_separator.separate_audio(
                                chunk_path, audio_sources
                            )
                        for i, audio_tensor in enumerate(chunk_separated):
                            if audio_tensor.dim() == 1:
                                audio_tensor = audio_tensor.unsqueeze(0)
                            stitched[i].append(audio_tensor.detach().cpu())

                    for i, parts in enumerate(stitched):
                        if len(parts) == 0:
                            continue
                        merged = torch.cat(parts, dim=-1)
                        save_path = os.path.join(save_dir, f"{prefix}{i}.wav")
                        audio_separator.save_audio(save_path, merged)


def embed_parsed_data(
    dataloader,
    feats_dir: str,
    embedder: CLAPEmbedder,
    seq_size: int = 20,
    text_parser_model: str = "gpt",
    audio_separator_model: str = "sam_audio",
):
    """
    Embed parsed audio segments and text prompts.
    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory to save features.
        embedder: CLAP embedder instance.
    """
    for batch in tqdm(dataloader, desc="Embedding Parsed Audio Segments"):
        text_ids = batch["text_id"]
        datasets = batch["dataset"]
        texts = batch["text"]
        rev_texts = batch["rev_text"]

        for text_id, dataset, text, rev_text in zip(
            text_ids, datasets, texts, rev_texts
        ):
            for text_str, suffix_text in [
                (text, ""),
                (rev_text, "_rev"),
            ]:
                if text_str == "":
                    continue

                # Load parsed audio sources
                text_path = os.path.join(
                    feats_dir,
                    f"{text_parser_model}_parsed_texts",
                    dataset,
                    f"{text_id + suffix_text}.json",
                )
                with open(text_path, "r", encoding="utf-8") as f:
                    audio_sources: list[str] = json.load(f)

                # Load separated audio files
                audio_dir = os.path.join(
                    feats_dir,
                    f"{audio_separator_model}_separated_audio",
                    dataset,
                    text_id + suffix_text,
                )

                all_audio_files = sorted(
                    [
                        os.path.join(audio_dir, f)
                        for f in os.listdir(audio_dir)
                        if f.endswith(".wav")
                    ]
                )
                rev_audio_files = [
                    file for file in all_audio_files if "rev_" in file.split("/")[-1]
                ]
                correct_audio_files = [
                    file for file in all_audio_files if file not in rev_audio_files
                ]

                for audio_files, prefix_audio in [
                    (correct_audio_files, ""),
                    (rev_audio_files, "rev_"),
                ]:
                    if len(audio_files) == 0:
                        continue
                    if not len(audio_files) == len(audio_sources):
                        raise ValueError(
                            f"Number of separated audio files ({len(audio_files)}) does not match number of audio sources ({len(audio_sources)}) for {text_id} in {dataset}."
                        )

                    # Embed audio and text
                    audio_embeddings = embedder.embed_audios(
                        audio_files
                    )  # [N * num_choices, D]
                    text_embeddings = embedder.embed_texts(audio_sources)  # [N, D]

                    num_segments = text_embeddings.shape[0]
                    embed_dim = text_embeddings.shape[-1]

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
                        audio_embeddings = torch.cat(
                            [audio_embeddings, audio_pad], dim=0
                        )
                        text_embeddings = torch.cat([text_embeddings, text_pad], dim=0)
                    else:
                        text_embeddings = text_embeddings[:seq_size]

                    # Save embeddings and mask
                    save_feats(
                        feats_dir=feats_dir,
                        feats_name=f"{embedder.name}_parsed_audio",
                        dataset=dataset,
                        file_name=f"{prefix_audio}{text_id}{suffix_text}.pt",
                        feats=audio_embeddings,
                    )
                    save_feats(
                        feats_dir=feats_dir,
                        feats_name=f"{embedder.name}_parsed_text",
                        dataset=dataset,
                        file_name=f"{prefix_audio}{text_id}{suffix_text}.pt",
                        feats=text_embeddings,
                    )
                    save_feats(
                        feats_dir=feats_dir,
                        feats_name=f"{embedder.name}_parsed_mask",
                        dataset=dataset,
                        file_name=f"{prefix_audio}{text_id}{suffix_text}.pt",
                        feats=mask,
                    )


def create_diff_audio(dataloader, feats_dir: str):
    """Create difference audio by subtracting separated audio from original.

    This function extracts residual sounds that are not captured in the
    separated audio files by subtracting the merged separated audio from
    the original audio.

    Args:
        dataloader: DataLoader for the dataset.
        feats_dir: Directory containing features and separated audio.
    """
    for batch in tqdm(dataloader, desc="Parsing Audio with SAM-Audio"):
        audio_files = batch["audio_file_path"]
        text_ids = batch["text_id"]
        datasets = batch["dataset"]

        for text_id, dataset, audio_file in zip(text_ids, datasets, audio_files):
            separated_audio_dir = os.path.join(
                feats_dir, "separated_audio", dataset, text_id
            )

            # Skip if separated audio directory doesn't exist
            if not os.path.exists(separated_audio_dir):
                continue

            separated_audio_files = sorted(
                [
                    os.path.join(separated_audio_dir, f)
                    for f in os.listdir(separated_audio_dir)
                    if f.endswith(".wav")
                ]
            )

            # Skip if no separated audio files exist
            if len(separated_audio_files) == 0:
                continue

            # Extract sounds not included in separated_audios
            # Take the value of separated_audio with maximum amplitude at each time step and subtract it from original_audio
            original_audio, orig_sr = torchaudio.load(audio_file)
            separated_audios = []
            for sep_audio_file in separated_audio_files:
                audio, sep_sr = torchaudio.load(sep_audio_file)
                separated_audios.append(audio)
            separated_audios = torch.stack(separated_audios, dim=0)  # (N, C, T)

            # Resample original audio to match separated audio sample rate
            if orig_sr != sep_sr:
                resampler = torchaudio.transforms.Resample(orig_sr, sep_sr)
                original_audio = resampler(original_audio)

            # Match lengths (truncate or pad to shorter length)
            orig_len = original_audio.shape[-1]
            sep_len = separated_audios.shape[-1]
            min_len = min(orig_len, sep_len)
            original_audio = original_audio[..., :min_len]
            separated_audios = separated_audios[..., :min_len]

            # Use the separated_audio value with maximum absolute value at each time step (waveform set)
            # This prevents phase inversion when the same sound is extracted multiple times
            abs_separated = torch.abs(separated_audios)  # (N, C, T)
            max_abs_idx = abs_separated.argmax(dim=0)  # (C, T)
            # Use gather to get the actual waveform values with maximum absolute value at each time step
            merged_separated = torch.gather(
                separated_audios, 0, max_abs_idx.unsqueeze(0)
            ).squeeze(0)  # (C, T)

            diff_audio = original_audio - merged_separated
            diff_audio = torch.clamp(diff_audio, -1.0, 1.0)

            # save diff audio
            diff_audio_dir = os.path.join(feats_dir, "diff_audio", dataset)
            os.makedirs(diff_audio_dir, exist_ok=True)
            diff_audio_path = os.path.join(diff_audio_dir, f"{text_id}.wav")
            torchaudio.save(diff_audio_path, diff_audio, sep_sr)


def embed_quality_prompts(feats_dir: str, embedder: CLAPEmbedder):
    """Embed PAM-style quality prompts and contrast prompts, save to disk.

    Args:
        feats_dir: Directory to save the quality prompt embeddings.
        embedder: CLAP embedder instance to use for text embedding.
    """

    QUALITY_PROMPTS = {
        "high": "the sound is clear and clean",
        "low": "the sound is noisy and with artifacts",
    }
    CONTRAST_PROMPTS = {
        "unrelated": "silence with no sound events",
    }

    save_dir = os.path.join(feats_dir, "quality_prompts")
    os.makedirs(save_dir, exist_ok=True)

    # Embed quality prompts (high/low)
    for quality_level, prompt in QUALITY_PROMPTS.items():
        save_path = os.path.join(save_dir, f"{quality_level}.pt")
        if os.path.exists(save_path):
            print(f"Quality prompt '{quality_level}' already exists at {save_path}")
            continue

        with torch.no_grad():
            embedding = embedder.embed_texts([prompt])  # [1, D]
        torch.save(embedding.squeeze(0), save_path)  # [D]
        print(f"Saved quality prompt '{quality_level}' to {save_path}")

    # Embed contrast prompts (unrelated) for DISCODE-style contrastive scoring
    for contrast_type, prompt in CONTRAST_PROMPTS.items():
        save_path = os.path.join(save_dir, f"{contrast_type}.pt")
        if os.path.exists(save_path):
            print(f"Contrast prompt '{contrast_type}' already exists at {save_path}")
            continue

        with torch.no_grad():
            embedding = embedder.embed_texts([prompt])  # [1, D]
        torch.save(embedding.squeeze(0), save_path)  # [D]
        print(f"Saved contrast prompt '{contrast_type}' to {save_path}")


def main(args):
    """
    Main preprocessing function.
    Args:
        args: Parsed command-line arguments.
    """
    embedder_model = args.clap_model
    text_parser_model = args.text_parser_model
    audio_separator_model = args.audio_separator_model

    if embedder_model == "humanclap":
        embedder = HumanCLAPEmbedder()
    elif embedder_model == "laionclap":
        embedder = LaionClapEmbedder()
    elif embedder_model == "msclap":
        embedder = MSClapEmbedder(seed=args.seed)

    # If only embedding quality prompts, do that and exit
    if args.quality_prompts:
        embed_quality_prompts(args.feats_dir, embedder)
        return

    for split in args.splits:
        dataset = TTAPreprocessDataset(data_dir=args.data_dir, split=split)
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

        clap_extract(dataloader, args.feats_dir, embedder=embedder)
        clear_gpu_memory()

        if text_parser_model == "gemini":
            text_parser = GeminiTextParser()
        elif text_parser_model == "gpt":
            text_parser = GPTTextParser()
        elif text_parser_model == "qwen":
            text_parser = QwenTextParser()

        text_parse(dataloader, args.feats_dir, text_parser=text_parser)
        del text_parser
        clear_gpu_memory()

        if audio_separator_model == "sam_audio":
            audio_separator = SAMAudioSeparator()
        elif audio_separator_model == "clapsep":
            audio_separator = CLAPSepSeparator()
        elif audio_separator_model == "soloaudio":
            audio_separator = SoloAudioSeparator()
        elif audio_separator_model == "audiosep":
            audio_separator = AudioSepSeparator()

        audio_separate(dataloader, args.feats_dir, audio_separator, text_parser_model)
        del audio_separator
        clear_gpu_memory()

        embed_parsed_data(
            dataloader,
            args.feats_dir,
            embedder,
            text_parser_model=text_parser_model,
            audio_separator_model=audio_separator_model,
        )
        clear_gpu_memory()

        create_diff_audio(dataloader, args.feats_dir)


### argument parser ###


def arg_parser():
    """Parse command-line arguments for preprocessing.

    Returns:
        Parsed argument namespace.
    """
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
        default=8,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to process (train/val/test)",
    )
    parser.add_argument(
        "--clap_model",
        type=str,
        default="humanclap",
        help="CLAP model to use (humanclap/laionclap/msclap)",
    )
    parser.add_argument(
        "--text_parser_model",
        type=str,
        default="gpt",
        help="LLM model to use for text parsing (gemini/gpt/qwen)",
    )
    parser.add_argument(
        "--audio_separator_model",
        type=str,
        default="sam_audio",
        help="Audio separator model to use (sam_audio/clapsep/soloaudio/audiosep)",
    )
    parser.add_argument(
        "--quality_prompts",
        action="store_true",
        help="Only embed quality prompts (skip other preprocessing)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    fix_seed(args.seed)
    main(args)
