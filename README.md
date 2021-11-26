# Wav2Vec2-Large-XLSR-53-Marathi
Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Marathi using the [Open SLR64](http://openslr.org/64/) dataset. When using this model, make sure that your speech input is sampled at 16kHz. This data contains only female voices but the model works well for male voices too. Trained on Google Colab Pro on Tesla P100 16GB GPU.<br>
**WER (Word Error Rate) on the Test Set**: 12.70 %
## Usage
The model can be used directly without a language model as follows, given that your dataset has Marathi `actual_text` and `path_in_folder` columns:
```python
import torch, torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#Since marathi is not present on Common Voice, script for reading the below dataset can be picked up from the eval script below
mr_test_dataset = all_data['test']
processor = Wav2Vec2Processor.from_pretrained("sumedh/wav2vec2-large-xlsr-marathi") 
model = Wav2Vec2ForCTC.from_pretrained("sumedh/wav2vec2-large-xlsr-marathi") 
resampler = torchaudio.transforms.Resample(48_000, 16_000) #first arg - input sample, second arg - output sample
# Preprocessing the datasets. We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
  speech_array, sampling_rate = torchaudio.load(batch["path_in_folder"])
  batch["speech"] = resampler(speech_array).squeeze().numpy()
  return batch
mr_test_dataset = mr_test_dataset.map(speech_file_to_array_fn)
inputs = processor(mr_test_dataset["speech"][:5], sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
  logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
predicted_ids = torch.argmax(logits, dim=-1)
print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", mr_test_dataset["actual_text"][:5])
```
## Evaluation
Evaluated on 10% of the Marathi data on Open SLR-64.
```python
import os, re, torch, torchaudio
from datasets import Dataset, load_metric
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#below is a custom script to be used for reading marathi dataset since its not present on the Common Voice
dataset_path = "./OpenSLR-64_Marathi/mr_in_female/" #TODO : include the path of the dataset extracted from http://openslr.org/64/
audio_df = pd.read_csv(os.path.join(dataset_path,'line_index.tsv'),sep='\t',header=None)
audio_df.columns = ['path_in_folder','actual_text']
audio_df['path_in_folder'] = audio_df['path_in_folder'].apply(lambda x: dataset_path + x + '.wav')
audio_df = audio_df.sample(frac=1, random_state=2020).reset_index(drop=True) #seed number is important for reproducibility of WER score
all_data = Dataset.from_pandas(audio_df)
all_data = all_data.train_test_split(test_size=0.10,seed=2020) #seed number is important for reproducibility of WER score
mr_test_dataset = all_data['test']
wer = load_metric("wer")
processor = Wav2Vec2Processor.from_pretrained("sumedh/wav2vec2-large-xlsr-marathi")
model = Wav2Vec2ForCTC.from_pretrained("sumedh/wav2vec2-large-xlsr-marathi") 
model.to("cuda")
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]' 
resampler = torchaudio.transforms.Resample(48_000, 16_000)
# Preprocessing the datasets. We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
  batch["actual_text"] = re.sub(chars_to_ignore_regex, '', batch["actual_text"]).lower()
  speech_array, sampling_rate = torchaudio.load(batch["path_in_folder"])
  batch["speech"] = resampler(speech_array).squeeze().numpy()
  return batch
mr_test_dataset = mr_test_dataset.map(speech_file_to_array_fn)
def evaluate(batch):
  inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
  return batch
result = mr_test_dataset.map(evaluate, batched=True, batch_size=8)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["actual_text"])))
```
## Training
Train-Test ratio was 90:10.
The training notebook Colab link [here](https://colab.research.google.com/drive/1wX46fjExcgU5t3AsWhSPTipWg_aMDg2f?usp=sharing).

## Training Config and Summary 
weights-and-biases run summary [here](https://wandb.ai/wandb/xlsr/runs/3itdhtb8/overview?workspace=user-sumedhkhodke)
