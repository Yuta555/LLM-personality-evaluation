# Personality detection model construction

We built classification models that detect/predict MBTI personality traits from text data by fine-tuning Llama 2 7B.

## 1. Preprocessing
Preprocess original dataset. (Kaggle MBTI data)
- Limit the number of tweets for each datapoint.
- Index and format input text data.

**How to create processed dataset** <br />
You just run the code below on terminal:
```
# This is the case using 25 tweets
python preprocess_data.py -num_tweets 25 -save_dir data/processed_data_25tweets
```

## 2. Training
Implement fine-tuning with LoRA on train/val dataset.
- We built two types of modes: 16-class classification model and binary classification model for each MBTI dimension.
- Before training, you may login to [Hugging Face](https://huggingface.co/) and [wandb](https://wandb.ai/site).
    ```
    # Login to Hugging Face
    huggingface-cli login --token $YOUR_HF_TOKEN

    # Login to wandb
    wandb login
    ```
- Run the script in the format below, then training process will start and three checkpoints and the final best model will be saved on output_dir.
    ```
    # Examples of scripts for fine-tuning
    python finetuning.py \
    --data_dir data/processed_data_50tweets \
    --lora_r 8 \
    --train_all_linear_layers False \
    --output_dir outputs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --save_name final_model \
    --dimension_id 0 \ 
    --wandb_project capstone-llama2-finetuning 
    ```
    - Omit dimension_id if 16-class model. Add 0 - 3 if a binary classification model.
    - Check and modify the python file for more detailed settings. For instance, we set gradient_checkpointing to reduce the GPU memory usage by fine-tuning so that the model can be trained on single L4 GPU (24GB memory), whereas it makes training speed slower.
 
You can see the results of fine-tuning such as train/eval loss and eval accuracy for each model [here](https://api.wandb.ai/links/ya2488/mkq940ni).

## 3. Prediction/Evaluation
Perform inference with fine-tuned model and evaluate model performance with accuracy score on test dataset.
- Evaluation on test dataset:
    ```
    python evaluation.py \
    --data_dir data/processed_data_50tweets \
    --checkpoint_dir outputs \ 
    --dimension_id 0

    # checkpoint_dir is directory where fine-tuned model is stored.
    ```

- You can use template notebook for prediction using fine-tuned models.
    - prediction_model_template.ipynb
