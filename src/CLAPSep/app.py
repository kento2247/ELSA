import gradio as gr
import librosa
import numpy as np
import torch

from model.CLAPSep import CLAPSep

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_path = "model/music_audioset_epoch_15_esc_90.14.pt"


model = CLAPSep(model_config, CLAP_path).to(DEVICE)
ckpt = torch.load("model/best_model.ckpt", map_location=DEVICE)
model.load_state_dict(ckpt, strict=False)
model.eval()


def inference(
    audio_file_path: str,
    text_p: str,
    audio_file_path_p: str,
    text_n: str,
    audio_file_path_n: str,
):
    # handling queries
    with torch.no_grad():
        embed_pos, embed_neg = torch.chunk(
            model.clap_model.get_text_embedding([text_p, text_n], use_tensor=True),
            dim=0,
            chunks=2,
        )
        embed_pos = torch.zeros_like(embed_pos) if text_p == "" else embed_pos
        embed_neg = torch.zeros_like(embed_neg) if text_n == "" else embed_neg
        embed_pos += (
            model.clap_model.get_audio_embedding_from_filelist([audio_file_path_p])
            if audio_file_path_p is not None
            else torch.zeros_like(embed_pos)
        )
        embed_neg += (
            model.clap_model.get_audio_embedding_from_filelist([audio_file_path_n])
            if audio_file_path_n is not None
            else torch.zeros_like(embed_neg)
        )

    print(
        f"Separate audio from [{audio_file_path}] with textual query p: [{text_p}] and n: [{text_n}]"
    )

    mixture, _ = librosa.load(audio_file_path, sr=32000)

    pad = (320000 - (len(mixture) % 320000)) if len(mixture) % 320000 != 0 else 0

    mixture = torch.tensor(np.pad(mixture, (0, pad)))

    max_value = torch.max(torch.abs(mixture))
    if max_value > 1:
        mixture *= 0.9 / max_value

    mixture_chunks = torch.chunk(mixture, dim=0, chunks=len(mixture) // 320000)
    sep_segments = []
    for chunk in mixture_chunks:
        with torch.no_grad():
            sep_segments.append(
                model.inference_from_data(chunk.unsqueeze(0), embed_pos, embed_neg)
            )

    sep_segment = torch.concat(sep_segments, dim=1)

    return 32000, sep_segment.squeeze().numpy()


with gr.Blocks(title="CLAPSep") as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Mixture", type="filepath")
            text_p = gr.Textbox(label="Positive Query Text")
            text_n = gr.Textbox(label="Negative Query Text")
            query_audio_p = gr.Audio(
                label="Positive Query Audio (optional)", type="filepath"
            )
            query_audio_n = gr.Audio(
                label="Negative Query Audio (optional)", type="filepath"
            )
        with gr.Column():
            with gr.Column():
                output_audio = gr.Audio(label="Separation Result", scale=10)
                button = gr.Button(
                    "Separate",
                    variant="primary",
                    scale=2,
                    size="lg",
                    interactive=True,
                )
                button.click(
                    fn=inference,
                    inputs=[input_audio, text_p, query_audio_p, text_n, query_audio_n],
                    outputs=[output_audio],
                )


demo.queue().launch(share=True)
