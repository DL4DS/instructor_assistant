from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import textstat
import pandas as pd

# Get questions and their metadata from threads
def get_questions(threads):
    step_question = []
    
    for thread in threads:
        for i in range(len(thread.steps)):
            if thread.steps[i].name == "on_message":
                step_question.append(thread.steps[i])
                
    dict = {
    'question': [list(d.input.values())[0] for d in step_question],
    'thread_id': [d.thread_id for d in step_question],
    'parent_id': [d.parent_id for d in step_question],
    'start_time': [d.start_time for d in step_question],
    }
    
    return dict

# Analyze question sentiment
def analyze_sentiment(questions):
    sentiment_data = []
    for question in questions:
        blob = TextBlob(question)
        sentiment = blob.sentiment
        
        sentiment_data.append({
            "Question": question,
            "Polarity": sentiment.polarity,
            "Subjectivity": sentiment.subjectivity
        })

    sentiment_df = pd.DataFrame(sentiment_data)

    return sentiment_df

# Extract keywords from questions
def extract_keywords(questions):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    all_words = []

    for question in questions:
        # Tokenize and lemmatize
        words = word_tokenize(question)
        words_filtered = [word.lower() for word in words if word.isalnum() and not word.isdigit()]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words_filtered]
        
        # Remove stopwords
        keywords = [word for word in lemmatized_words if word not in stop_words]
        all_words.extend(keywords)

    most_common_keywords = Counter(all_words).most_common()
    print("Most common keywords and their frequencies:")
    for keyword, frequency in most_common_keywords:
        print(f"{keyword}: {frequency}")


# Analyze question complexity
def analyze_complexity(questions):
    complexity = []

    for question in questions:
        # Flesch Reading Ease - readability (higher = simpler)
        flesch_reading_ease = textstat.flesch_reading_ease(question)
        
        # Gunning Fog index - years of formal education (higher = more complex)
        gunning_fog = textstat.gunning_fog(question)
        
        # Avg number of syllables per word
        avg_syllables_per_word = textstat.syllable_count(question) / len(question.split())
        

        complexity.append({
            "Question": question,
            "Flesch Reading Ease": flesch_reading_ease,
            "Gunning Fog Index": gunning_fog,
            "Avg Syllables": avg_syllables_per_word
        })

    complexity_df = pd.DataFrame(complexity)
    
    return complexity_df
