import torch
import torchaudio
import soundfile as sf

def main():
    input_path = "TP3/data/call_01.wav"
    output_path = "TP3/data/call_01.wav"
    
    print(f"Loading {input_path}...")
    data, sr = sf.read(input_path)
    wav = torch.from_numpy(data).float()
    
    # Handle channels
    if wav.ndim == 1:
        wav = wav.unsqueeze(0) # [1, time]
    else:
        wav = wav.t() # [channels, time]
        
    # Mix to mono
    wav = wav.mean(dim=0, keepdim=True)
    
    # Resample
    target_sr = 16000
    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    
    # Add simulated microphone noise
    print("Adding microphone noise...")
    noise_level = 0.005
    noise = torch.randn_like(wav) * noise_level
    wav = wav + noise
    
    # Normalize to prevent clipping
    max_val = wav.abs().max()
    if max_val > 0.99:
        wav = wav / max_val * 0.99
    
    # Save
    print(f"Saving to {output_path}...")
    # standardizing to [-1, 1] is good practice but let's just save
    sf.write(output_path, wav.squeeze().numpy(), target_sr)
    print("Done.")

if __name__ == "__main__":
    main()
