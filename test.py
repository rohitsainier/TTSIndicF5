from time import time
from transformers import AutoModel
import numpy as np
import soundfile as sf
import json
from datetime import datetime

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