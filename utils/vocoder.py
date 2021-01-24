import torch

from vocoder.vocgan.generator import Generator
from scipy.io import wavfile
from config import Arguments as args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vocgan(ckpt_path, n_mel_channels=args.n_mels, generator_ratio = [4, 4, 2, 2, 2, 2], n_residual_layers=4, mult=256, out_channels=1):

    checkpoint = torch.load(ckpt_path)
    model = Generator(n_mel_channels, n_residual_layers,
                        ratios=generator_ratio, mult=mult,
                        out_band=out_channels)

    model.load_state_dict(checkpoint['model_g'])
    model.to(device).eval()

    return model

def vocgan_infer(mel, vocoder, path):
    model = vocoder

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)

        audio = model.infer(mel).squeeze()
        audio = args.max_wav_value * audio[:-(args.hop_length*10)]
        audio = audio.clamp(min=-args.max_wav_value, max=args.max_wav_value-1)
        audio = audio.short().cpu().detach().numpy()

        wavfile.write(path, args.sr, audio)
