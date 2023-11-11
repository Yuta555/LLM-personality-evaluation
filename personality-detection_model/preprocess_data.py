import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict


SEED = 42
NUM_TWEETS = 25
original_data = 'data/kaggle.csv'
save_dir = 'data/processed_data_no_removal'


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


if __name__=="__main__":
    # Load Kaggle dataset for MBTI
    df = pd.read_csv(original_data)

    # Preprocess text data
    df['text'] = df['text'].apply(lambda x: limit_tweets(x, NUM_TWEETS)) # Decrease the number of tweets for each sample by 25
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