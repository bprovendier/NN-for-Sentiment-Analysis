"""Module processing and cleaning the data as well as performing sentiment analysis."""
import pandas as pd
import numpy as np
import torch
import os
from datetime import timedelta, datetime
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def load_file(directory, file_name):
    """
    File loading function.

    Arguments:
        directory (Literal): name of the directory.
        file_name (str): name of the file.

    Returns:
        headlines_df (DataFrame): content within csv file.
    """
    headlines_df = pd.read_csv(f"{directory}/{file_name}")
    headlines_df = headlines_df.dropna()

    return headlines_df


def clean_df(headlines_df, file_name):
    """
    Clean tweet text.

    Arguments:
        headlines_df (DataFrame): tweet information dataframe.
        file_name (str): name of the file.

    Returns:
        original_array (ndarray): array contaning uncleaned tweet text.
        headlines_array (ndarray): array contaning cleaned tweet text.
    """
    # only take tweets that were published 15 min before collection
    title_time = re.search(r"\d{2}:\d{2}", headlines["Date"][0])
    end_time = datetime.strptime(title_time.group(), "%H:%M")
    start_time = end_time - timedelta(hours=0, minutes=15)

    text = []
    for i in range(len(headlines_df["Date"])):
        try:
            tweet = re.search(r"\d{2}:\d{2}", headlines_df["Date"][i])
            tweet_time = datetime.strptime(tweet.group(), "%H:%M")
            if tweet_time.time() >= start_time.time():
                cleaned = text_preprocessing(headlines_df["Tweet Text"][i])
                text.append([headlines_df["Tweet Text"][i], cleaned])
        except Exception as e:
            print("Error", e, "  in file: ", file_name, " for lines number: ", i)
            pass
    new_df = pd.DataFrame(text, columns=["Raw Tweets", "Cleaned Tweets"])
    new_df = new_df.dropna()
    new_df = new_df.drop_duplicates(subset=["Cleaned Tweets"])

    original_array = np.array(new_df["Raw Tweets"])
    headlines_array = np.array(new_df["Cleaned Tweets"])

    return original_array, headlines_array


contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
}


def text_preprocessing(text):
    """
    Clean text tweet by tweet.

    Arguments:
        text (str): individual uncleaned tweet.

    Returns:
        text (str): cleaned tweet.
    """
    # Convert words to lower case
    text = text.lower()

    # Remove @name
    text = re.sub(r"\S*@?:\S*", "", text)
    # Remove links
    text = re.sub(r"\S*http?:\S*", "", text)
    text = re.sub(r"\S*https?:\S*", "", text)
    # Remove hashtags
    text = re.sub(r"\S*#?:\S*", "", text)

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r"&amp;", "", text)
    text = re.sub(r"0,0", "00", text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', "", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"\$", " $ ", text)
    text = re.sub(r"u s ", "united states", text)
    text = re.sub(r"u n ", " united nations", text)
    text = re.sub(r"u k ", "united kingdom ", text)
    text = re.sub(r"j k ", "jk", text)
    text = re.sub(r" s ", " ", text)
    text = re.sub(r" yr ", "year", text)
    text = re.sub(r" l g b t ", "lgbt", text)
    text = re.sub(r"0km ", "0 km", text)

    # Change 't to 'not'
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\’t", " not", text)
    # Isolate and remove punctuations except '?'
    text = re.sub(r"([\'\"\.\(\)\!\?\\\/\,])", r"", text)
    text = re.sub(r"[^\w\s\?]", "", text)
    # Remove some special characters
    text = re.sub(r"([\;\:\|#•«\n])", "", text)
    # Remove trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def model_evaluation(tweet):
    """
    Sentiment analysis function.

    Arguments:
        tweet (str): individual tweet.

    Returns:
        predictions
    """
    try:
        inputs = tokenizer(tweet, padding=True, truncation=True, return_tensors="pt")
    except:
        print("This tweet did not tokenise correctly: ", tweet)
        fake_tweet = ["Neutral"]
        inputs = tokenizer(
            fake_tweet, padding=True, truncation=True, return_tensors="pt"
        )
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    return predictions


def sentiment_score(model_result):
    """
    Get comprehensive sentiment score results.

    Arguments:
        model_result

    Returns:
        table (list): list with positive, neutral and negative scores.
    """
    model.config.id2label
    positive = model_result[:, 0].tolist()
    negative = model_result[:, 1].tolist()
    neutral = model_result[:, 2].tolist()

    table = [
        positive[0],
        neutral[0],
        negative[0],
    ]

    return table


def get_sentiment_score(table_score):
    """
    Calculate sentiment score.

    Arguments:
        table_score (list): list with positive, neutral and negative scores.

    Returns:
        score (float): sentiment score.
    """
    score = table_score[0] - table_score[2]

    return score


def number_positive(scores):
    """
    Calculate the number of positive tweets.

    Arguments:
        scores (list): list of scores in file.

    Returns:
        pos (int): number of positive tweets.
    """
    pos = 0
    # we choose 0.2 as a trigger value instead of 0 to get clear positivity
    for value in scores:
        if value >= 0.2:
            pos += 1

    return pos


def classify_score(total, sentiment_score):
    """
    Update overall total file sentiment score.

    Arguments:
        total (int): total sentiment score.
        sentiment_score (list): list with positive, neutral and negative scores.

    Returns:
        total (int): updated sentiment score.
    """
    if (sentiment_score[0] > sentiment_score[2]) & (
        sentiment_score[0] >= sentiment_score[1]
    ):
        total += 1
    elif (sentiment_score[2] > sentiment_score[0]) & (
        sentiment_score[2] >= sentiment_score[1]
    ):
        total -= 1

    return total


bitcoin = "BTC"
ethereum = "ETH"
dir = "../"

file_specific_columns = [
    "Date",
    "Time",
    "Original tweet",
    "Clean tweet",
    "Positive",
    "Neutral",
    "Negative",
]
master_columns = [
    "Date",
    "Time",
    "Number of tweets",
    "Additive score",
    "Mean score",
    "Standard deviation score",
    "Number of positive",
    "Percentage of positive",
]

keyword = bitcoin
file_path_name = f"../file_sentiment_{keyword}"

master_path = r"../"
start_date_time = datetime.strptime("2000-00-00_00-00", "%Y-%m-%d_%H-%M")


for fname in os.listdir(dir):
    if keyword in fname:
        file_scores = []
        total_file_score = 0
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", fname)
        dates = datetime.strptime(match.group(), "%Y-%m-%d_%H-%M")

        # account for UK daylight saving time change
        summer_time = "2022-03-27_01-00"
        summer_dates = datetime.strptime(summer_time, "%Y-%m-%d_%H-%M")
        if dates > summer_dates:
            dates = dates - timedelta(hours=1)

        date = f"{dates.date()}"
        time = f"{dates.hour}:{dates.minute}"
        time_title = f"{dates.hour}-{dates.minute}"
        if dates <= start_date_time:
            print(date, time)
            pass
        else:
            print(date, time)
            try:
                file_specific_data = []
                headlines = load_file(dir, fname)
                original_headlines, clean_headlines = clean_df(headlines, fname)

                for i in range(0, len(clean_headlines)):
                    tweet = original_headlines[i]
                    clean_tweet = clean_headlines[i]
                    result = model_evaluation(clean_tweet)
                    score_table = sentiment_score(result)
                    file_specific_data.append(
                        [
                            date,
                            time,
                            tweet,
                            clean_tweet,
                            score_table[0],
                            score_table[1],
                            score_table[2],
                        ]
                    )

                    tweet_score = get_sentiment_score(score_table)
                    file_scores.append(tweet_score)
                    total_file_score = classify_score(total_file_score, score_table)

                file_specific_df = pd.DataFrame(
                    file_specific_data, columns=file_specific_columns
                )
                file_name = f"TweetSentiment_{keyword}_{date}_{time_title}.csv"
                file_specific_df.to_csv(Path(file_path_name, file_name), index=False)

                positive = number_positive(file_scores)
                final_scores = pd.Series(file_scores)
                percentage_pos = positive / len(final_scores)
                master_df = pd.read_csv(
                    Path(master_path, f"master_tweet_sentiment_{keyword}.csv")
                )
                new_data = [
                    [
                        date,
                        time,
                        len(clean_headlines),
                        total_file_score,
                        final_scores.mean(),
                        final_scores.std(),
                        positive,
                        percentage_pos,
                    ]
                ]
                new_line = pd.DataFrame(new_data, columns=master_columns)
                master_df = master_df.append(new_line)
                master_df.to_csv(
                    Path(master_path, f"master_tweet_sentiment_{keyword}.csv"),
                    index=False,
                )
            except:
                print("SKIPPING FILE", fname)
                pass
