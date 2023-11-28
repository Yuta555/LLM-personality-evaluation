import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict

os.makedirs('data/', exist_ok=True)

SEED = 42

# limit_tweets
def limit_tweets(tweets, num_tweets):
    tweets = tweets.split(sep='|||') # Each tweet are separated by "|||"
    tweets = tweets[:num_tweets]
    tweets = '|||'.join(tweets)
    return tweets

# Indexing tweets, assuming each tweet is separated by "|||"
def index_tweets(tweets):
    sep_tweets = tweets.split("|||")
    
    indexed_tweets = []    
    for i, tweet in enumerate(sep_tweets):
        indexed_tweets.append(f"{i+1}. {tweet}")
    
    indexed_tweets = "\n".join(indexed_tweets)
    
    return indexed_tweets
            
# Save preprocessed data
def save_preprocessed_dataset(path: str, dataset):
    dataset.save_to_disk(path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-num_tweets", type=int, default=25)
    ap.add_argument("-original_data", type=str, default="data/kaggle.csv")
    ap.add_argument("-save_dir", type=str, default="data/processed_data")
    args = ap.parse_args()
    return (
        args.num_tweets,
        args.original_data,
        args.save_dir
    )


if __name__=="__main__":
    (
        num_tweets,
        original_data,
        save_dir,
    ) = parse_args()


    # Load Kaggle dataset for MBTI
    df = pd.read_csv(original_data)

    # Preprocess text data
    df['text'] = df['text'].apply(lambda x: limit_tweets(x, num_tweets)) # Decrease the number of tweets for each sample by 25
    df['text'] = df['text'].apply(lambda x: index_tweets(x))

    # Split into train/test dataset
    text_train, text_test, label_train, label_test = train_test_split(df['text'], df['type'], test_size=0.1, random_state=SEED)

    # Store data in DatasetDict
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_dict(
        {"text": text_train, "label": label_train}
    )
    dataset["test"] = Dataset.from_dict(
        {"text": text_test, "label": label_test}
    )

    save_preprocessed_dataset(save_dir, dataset)