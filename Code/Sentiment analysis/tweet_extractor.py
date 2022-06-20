"""Twitter data collection sent via email."""
import tweepy
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from credentials import twitter_credentials, google_credentials


class Error(Exception):
    """Base class for other exceptions."""

    pass


class TweepError(Error):
    """Twitter API connection error class."""

    pass


def api_connection():
    """
    Twitter API connection function.

    Returns:
        api (API): api connection token.
    """
    credentials = twitter_credentials()

    consumer_key = credentials[0]
    consumer_secret = credentials[1]
    access_token = credentials[2]
    access_secret = credentials[3]

    authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
    authenticate.set_access_token(access_token, access_secret)
    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    return api


def gather_tweets(crypto_search, long_name, number_of_tweets):
    """
    Twitter API request function.

    Arguments:
        crypto_search (str): crypto ticker symbol.
        long_name (str): crypto long name.
        number_of_tweets (int): specify the number of tweets to request.

    Returns:
        tweets (ItemIterator): json iterator with tweets.
    """
    api = api_connection()
    search_term = f"#{crypto_search} OR #{long_name} -filter:retweets -filter:replies"
    max_tweet = number_of_tweets

    try:
        tweets = tweepy.Cursor(
            api.search_tweets,
            q=search_term,
            lang="en",
            tweet_mode="extended",
            result_type="recent",
        ).items(max_tweet)
    except:
        error = "API error. Could be due to limit usage"
        error_email(error)
        print(error)

    return tweets


def create_tweet_df(tweets, name_crypto):
    """
    Create structure dataframe with tweets and the relevant informations.

    Arguments:
        tweets (ItemIterator): json iterator with tweets.
        name_crypto (str): crypto ticker symbol.
    """
    date = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_info = [[] for _ in range(7)]
    try:
        for tweet in tweets:
            new_tweet = json.dumps(tweet._json)
            dct = json.loads(new_tweet)
            new_info = [
                dct["created_at"],
                dct["full_text"],
                dct["retweet_count"],
                dct["favorite_count"],
                dct["user"]["screen_name"],
                dct["user"]["followers_count"],
                dct["id"],
            ]
            for i in range(len(all_info)):
                all_info[i].append(new_info[i])
    except TweepError:
        error = "Twitter Credentials Error. Could not connect to API."
        error_email(error, date)
        print(error)
    except:
        error = "Unkown Error. Could not gather tweets."
        error_email(error, date)
        print(error)

    latest_df = pd.DataFrame(all_info).transpose()
    latest_df.columns = [
        "Date",
        "Tweet Text",
        "Retweet Count",
        "Favorite Count",
        "Username",
        "User follower count",
        "ID",
    ]
    name_file = f"{date}_{name_crypto}_twitter_data.csv"
    latest_df.to_csv(f"data/{name_file}")

    send_email(date, name_file)
    [f.unlink() for f in Path("data/").glob("*") if f.is_file()]

    return 0


def send_email(date, file_name):
    """
    Send tweet dataframe by email.

    Arguments:
        date (str): date and time used for file name.
        file_name (str): name of the file to load and attach in email.
    """
    email_credentials = google_credentials()
    subject = f"Twitter Data {date}"

    sender_email = email_credentials[0]
    receiver_email = email_credentials[0]
    password = email_credentials[1]

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    with open(f"data/{file_name}", "rb") as file:
        message.attach(MIMEApplication(file.read(), Name=f"{file_name}"))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())


def error_email(error_message, date):
    """
    Error email function in case of error.

    Argument:
        error_message (Literal): error as email body.
        date (str): date and time of the day for email subject.
    """
    email_credentials = google_credentials()
    subject = f"[URGENT] TWITTER DATA ERROR {date}"

    sender_email = email_credentials[0]
    receiver_email = email_credentials[0]
    password = email_credentials[1]

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    html = f"""\
    <html>
    <head></head>
    <body>
        {error_message}
    </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())


cryptos = [
    "BTC",
    "ETH",
]
real_name = ["Bitcoin", "Ethereum"]
num_tweet = 300

for i in range(len(cryptos)):
    all_tweets = gather_tweets(cryptos[i], real_name[i], num_tweet)
    create_tweet_df(all_tweets, cryptos[i])
