# **IndicF5: High-Quality TextTo gTo ge22. **A reference voice audio\*\* – An example speech clip that guides the model's prosody and speaker characteristics.

3. **Text spoken in the reference voice audio** – The transcript of the reference voice audio.**A reference voice audio** – An example speech clip that guides the model's prosody and speaker characteristics.
4. **Text spoken in the reference voice audio** – The transcript of the reference voice audio.rate speech, you need to provide **three inputs**:
5. **Text to synthesize** – The content you want the model to speak.
6. **A reference voice audio** – An example speech clip that guides the model's prosody and speaker characteristics.
7. **Text spoken in the reference voice audio** – The transcript of the reference voice audio.**A reference voice audio** – An example speech clip that guides the model's prosody and speaker characteristics.
8. **Text spoken in the reference voice audio** – The transcript of the reference voice audio.rate speech, you need to provide **three inputs**:
9. **Text to synthesize** – The content you want the model to speak.
10. **A reference voice audio** – An example speech clip that guides the model's prosody and speaker characteristics.
11. **Text spoken in the reference voice audio** – The transcript of the reference voice audio.Speech for Indian Languages\*\*

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)

We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), [IndicTTS](https://www.iitm.ac.in/donlab/indictts/database), [LIMMITS](https://sites.google.com/view/limmits24/), and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)**.

IndicF5 supports **11 Indian languages**:  
**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**

---

## 🚀 Installation

```bash
conda create -n indicf5 python=3.12 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/IndicF5.git
```

- or

```sh
python3 -m venv .venv
source .venv/bin/activate

# pip install -e .
pip install -r requirements.txt
```

## 🎙 Usage

To generate speech, you need to provide **three inputs**:

1. **Text to synthesize** – The content you want the model to speak.
2. **A reference prompt audio** – An example speech clip that guides the model’s prosody and speaker characteristics.
3. **Text spoken in the reference prompt audio** – The transcript of the reference prompt audio.

- just run `./start_api.sh` or `HF_TOKEN=XXX ./start_api.sh` for login to HF for downloading Model
- run `test.py` for quick test
- update the `reference_voices.json` file path and `reference_voice_key` and `text_to_convert` to desired Indic Language
- output will be generated in `data/out/*.wav` (Note: generate the folders if not available)

```sh
python test.py
```

- or create a new python script

```python
from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "hareeshbabu82/TeluguIndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
    ref_audio_path="reference_voices/PAN_F_HAPPY_00001.wav",
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
```

## 🏷️ Prompt Tag Processing

IndicF5 now supports processing text with multiple voice segments using prompt tags. This allows you to create combined audio with different voices and referenceVoices in a single call.

### Format

Use the following XML-like format in your text:

```xml
<refvoice key="reference_voice_key">text to speak with this prompt key</refvoice>
```

### Example Usage

```python
from tts_utils import generate_speech_from_reference_voice_tags

# Text with multiple voice segments
text = '''
<refvoice key="TEL_F_WIKI_00001">ఇది తెలుగు లో మాట్లాడుతోంది.</refvoice>

<refvoice key="HIN_F_HAPPY_00001">यह हिंदी में बोल रहा है।</refvoice>

<refvoice key="PAN_F_HAPPY_00001">ਇਹ ਪੰਜਾਬੀ ਵਿੱਚ ਬੋਲ ਰਿਹਾ ਹੈ।</refvoice>
'''

# Generate combined audio
result = generate_speech_from_reference_voice_tags(
    text=text,
    output_path="multilingual_output.wav",
    pause_duration=300  # 300ms pause between segments
)

if result['success']:
    print(f"Generated audio with {result['segments_processed']} segments")
    print(f"Saved to: {result['output_path']}")
```

### Base Prompt Key Feature

You can now specify a `base_reference_voice_key` to handle text outside of prompt tags:

```python
# Text with mixed tagged and untagged content
text = '''
కాకులు ఒక పొలానికి వెళ్లి అక్కడ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి.

<refvoice key="TEL_M_WIKI_00001">अब मैं हिंदी में बोल रहा हूं।</refvoice>

తిరుమల వెంకటేశ్వర ఆలయం ప్రపంచంలో అత్యధికంగా సందర్శించే పుణ్యక్షేత్రాలలో ఒకటి.

<refvoice key="TEL_F_WIKI_00001">ఇప్పుడు తెలుగులో మాట్లాడుతున్నాను.</refvoice>

ఊరేగింపుకు ఒక ఎద్దు బండి కట్టేవారు. ఆ బండిని కడిగి, పసుపు రాసి, బొట్లు పెట్టి, పూలు కట్టి దాన్ని కూడా అందంగా అలంకరించేవారు.
'''

# Generate with base prompt for untagged text
result = generate_speech_from_reference_voice_tags(
    text=text,
    base_reference_voice_key="PAN_F_HAPPY_00001",  # Voice for English parts
    output_path="mixed_content_output.wav",
    pause_duration=250
)
```

### Advanced Usage

```python
from tts_utils import TTSProcessor

# Create processor
processor = TTSProcessor()
processor.load_model()
processor.load_reference_voices()

# Process with custom settings
result = processor.process_reference_voice_tagged_text(
    text=text,
    base_reference_voice_key="TEL_F_WIKI_00001",  # Base voice for untagged content
    output_path="output.wav",
    sample_rate=24000,
    pause_duration=500,     # 500ms pause between segments
    max_chunk_chars=200,    # Split long texts into chunks
    normalize=True
)
```

### Features

- **Multi-voice support**: Use different prompt keys for different voice characteristics
- **Base prompt key**: Specify a default voice for text outside of prompt tags
- **Automatic audio combination**: Seamlessly combines segments with customizable pauses
- **Text chunking**: Automatically handles long text segments
- **Error handling**: Validates prompt keys and provides detailed error messages
- **Batch processing**: Processes multiple segments efficiently

### Available Test Scripts

- `example_reference_voice_tags.py` - Basic example of prompt tag usage
- `example_enhanced_reference_voice_tags.py` - Advanced examples with base prompt key
- `test_reference_voice_tags.py` - Comprehensive test suite with error handling
- `test_base_prompt.py` - Test base prompt key functionality

## References

We would like to extend our gratitude to the authors of **[F5-TTS](https://github.com/SWivid/F5-TTS)** for their invaluable contributions and inspiration to this work. Their efforts have played a crucial role in advancing the field of text-to-speech synthesis.

## 📖 Citation

If you use **IndicF5** in your research or projects, please consider citing it:

### 🔹 BibTeX

```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}
```
