# Personality detection model construction

We built classification models that detect/predict MBTI personality traits from text data by fine-tuning Llama 2 7B.

## 1. Preprocessing
Preprocess original dataset. (Kaggle MBTI data) (preprocess_data.py)
- Limit the number of tweets for each datapoint.
- Index and format input text data.


## 2. Training
Implement fine-tuning with train/val dataset.
- LoRA (lora_tuning.ipynb)
- Adapter (XXX)

## 3. Inference/Evaluation
Perform inference with fine-tuned model and evaluate model performance with accuracy score on test dataset. (inference_after_lora.ipynb)
