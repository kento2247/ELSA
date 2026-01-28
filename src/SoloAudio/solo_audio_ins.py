import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel

from SoloAudio.model.udit import UDiT
from SoloAudio.vae_modules.autoencoder_wrapper import Autoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def sample_diffusion(
    unet,
    autoencoder,
    scheduler,
    mixture,
    timbre,
    device,
    ddim_steps=50,
    eta=0,
    seed=42,
):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    # init noise
    noise = torch.randn(mixture.shape, generator=generator, device=device)
    pred = noise

    for t in scheduler.timesteps:
        pred = scheduler.scale_model_input(pred, t)
        model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
        pred = scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=pred,
            eta=eta,
            generator=generator,
        ).prev_sample

    pred = autoencoder(embedding=pred).squeeze(1)

    return pred


class SoloAudio(nn.Module):
    def __init__(
        self,
        clap_model_path: str = "laion/larger_clap_general",
        autoencoder_path: str = "models/soloaudio_vae.pt",
        soloaudio_path: str = "models/soloaudio.pt",
    ):
        super().__init__()
        self.clapmodel = ClapModel.from_pretrained(clap_model_path).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(clap_model_path)
        udit_config = {
            "input_dim": 256,
            "output_dim": 128,
            "pos_method": "none",
            "pos_length": 500,
            "timbre_dim": 512,
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
        }
        self.autoencoder = Autoencoder(
            autoencoder_path, "stable_vae", quantization_first=True
        )
        self.unet = UDiT(**udit_config).to(DEVICE)
        self.unet.load_state_dict(torch.load(soloaudio_path)["model"])

        diffuser_config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
            "clip_sample": False,
        }
        self.noise_scheduler = DDIMScheduler(**diffuser_config)

        # these steps reset dtype of noise_scheduler params
        latents = torch.randn((1, 128, 128), device=DEVICE)
        noise = torch.randn(latents.shape).to(DEVICE)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (noise.shape[0],),
            device=DEVICE,
        ).long()
        _ = self.noise_scheduler.add_noise(latents, noise, timesteps)

    def forward(
        self,
        audio_tensor: torch.Tensor,
        prompt: str,
    ):
        audio_tensor = self.autoencoder(audio=audio_tensor.unsqueeze(1))
        text_inputs = self.processor(
            text=[prompt],
            max_length=10,  # Fixed length for text
            padding="max_length",  # Pad text to max length
            truncation=True,  # Truncate text if it's longer than max length
            return_tensors="pt",
        )
        inputs = {
            "input_ids": text_inputs["input_ids"][0].unsqueeze(0),  # Text input IDs
            "attention_mask": text_inputs["attention_mask"][0].unsqueeze(
                0
            ),  # Attention mask for text
        }
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        timbre = self.clapmodel.get_text_features(**inputs)

        pred = sample_diffusion(
            self.unet,
            self.autoencoder,
            self.noise_scheduler,
            audio_tensor,
            timbre,
            DEVICE,
            ddim_steps=50,
            eta=0,
            seed=42,
        )

        return pred
