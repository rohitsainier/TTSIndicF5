from time import time
from transformers import AutoModel
import numpy as np
import soundfile as sf
import json
from datetime import datetime
from importlib.resources import files
from f5_tts.api import F5TTS

# Global variables
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = None

start_time = time()

# Load INF5 from Hugging Face
repo_id = "hareeshbabu82/TeluguIndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
end_time = time()
print(f"Model loaded in {end_time - start_time:.2f} seconds")

# load reference voices from data/reference_voices/reference_voices.json

with open("data/reference_voices/reference_voices.json", "r", encoding="utf-8") as f:
    reference_voices = json.load(f)

reference_voice_key = "PAN_F_HAPPY_00001"
text_to_convert = "కాకులు ఒక పొలానికి వెళ్లి అక్కడ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి. పిచుక నిస్సహాయంగా ఏమి చేయాలో తెలీకా అటూ ఇటూ గెంతుతూ వుంది. ఇంతలో ఆ పొలం రైతులు పరిగెత్తుకుంటూ వచ్చి ఒక పెద్ద కర్రతో ఆ కాకులను కొట్టడం మొదలెట్టారు. కాకుల గుంపుకు ఇది అలవాటే, అవి తుర్రున ఎగిరిపోయాయి. పిచుక రైతులకు దొరికిపోయింది."

start_time = time()
# Generate speech
audio = model(
    text_to_convert,
    ref_audio_path=reference_voices[reference_voice_key]["file"],
    ref_text=reference_voices[reference_voice_key]["content"],
)
end_time = time()
print(f"Speech generation took {end_time - start_time:.2f} seconds")

start_time = time()
# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sf.write(f"data/out/gen_{timestamp}.wav", np.array(audio, dtype=np.float32), samplerate=24000)
end_time = time()
print(f"Audio saved in {end_time - start_time:.2f} seconds")



f5tts = F5TTS(model="F5TTS_Base")
start_time = time()
wav, sr, spec = f5tts.infer(
    ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="""Berlin ist nicht nur Weltmetropole und die Hauptstadt Deutschlands, sondern auch meine Heimatstadt.""",
    # gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
    file_wave=str(files("data").joinpath(f"out/gen_f5tts_{timestamp}.wav")),
    # file_spec=str(files("data").joinpath("out/api_out.png")),
    # seed=4,
)
end_time = time()
print(f"F5TTS Audio saved in {end_time - start_time:.2f} seconds")