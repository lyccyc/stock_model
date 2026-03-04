import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Set device to CPU explicitly if no GPU is available
device = 0 if torch.cuda.is_available() else -1

print("Initializing Translation Model (Helsinki-NLP/opus-mt-zh-en)...")
translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
if device == 0:
    translation_model = translation_model.to("cuda")

print("Initializing Sentiment Analysis Model (ProsusAI/finbert)...")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("text-classification", model=finbert_model, tokenizer=finbert_tokenizer, device=device, top_k=None)

def translate_to_english(text):
    """
    Translates Traditional Chinese text to English
    """
    if not text or pd.isna(text):
        return ""
    try:
        inputs = translation_tokenizer(text[:512], return_tensors="pt")
        if device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = translation_model.generate(**inputs)
        return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def analyze_sentiment(english_text):
    """
    Passes translated english text through FinBERT.
    Extracts positive, negative, neutral probabilities.
    Returns calculated daily_sentiment_index = (pos - neg).
    """
    if not english_text:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'sentiment_index': 0.0}
    
    try:
        # Ensure we don't overflow the tokenizer
        results = sentiment_pipeline(english_text[:512])[0]
        
        # Mapping results
        scores = {res['label']: res['score'] for res in results}
        
        pos = scores.get('positive', 0.0)
        neg = scores.get('negative', 0.0)
        neu = scores.get('neutral', 0.0)
        
        # Calculate daily_sentiment_index from -1.0 to 1.0
        daily_sentiment_index = pos - neg
        
        return {
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'sentiment_index': daily_sentiment_index
        }
    except Exception as e:
        print(f"Sentiment evaluation error: {e}")
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'sentiment_index': 0.0}

def process_news():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    raw_news_path = os.path.join(data_dir, 'news_raw.csv')
    
    if not os.path.exists(raw_news_path):
        print(f"Could not find raw news at {raw_news_path}. Did you run news_collector.py?")
        return
        
    print("Loading raw news data...")
    df = pd.read_csv(raw_news_path)
    if df.empty:
        print("Raw news data is empty.")
        return
        
    print(f"Processing {len(df)} news items...")
    results = []
    for idx, row in df.iterrows():
        headline = str(row.get('title', ''))
        snippet = str(row.get('snippet', ''))
        date_str = str(row.get('date', ''))
        
        # Compile full contextual string
        full_text = f"{headline}. {snippet}"
        
        # 2. Translation Layer
        translated_text = translate_to_english(full_text)
        
        # 3. Sentiment Analysis (FinBERT)
        sentiment_scores = analyze_sentiment(translated_text)
        
        results.append({
            'date': date_str,
            'original_title': headline,
            'translated_text': translated_text,
            'positive': sentiment_scores['positive'],
            'negative': sentiment_scores['negative'],
            'neutral': sentiment_scores['neutral'],
            'sentiment_index': sentiment_scores['sentiment_index']
        })
        
    # 5. Data Integration & Export
    results_df = pd.DataFrame(results)
    output_path = os.path.join(data_dir, 'news_sentiment_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Processed sentiment analysis results successfully saved to: {output_path}")

if __name__ == "__main__":
    # 6. Validation Layer
    print("--- Running Test Scenario ---")
    test_headline = "台積電營收超預期，股價看漲"
    print(f"Source Text: {test_headline}")
    
    translated_test = translate_to_english(test_headline)
    print(f"Translated Text: {translated_test}")
    
    scores = analyze_sentiment(translated_test)
    print(f"FinBERT Softmax -> Positive: {scores['positive']:.3f} | Negative: {scores['negative']:.3f} | Neutral: {scores['neutral']:.3f}")
    print(f"Daily Sentiment Index: {scores['sentiment_index']:.3f} (Range: -1.0 to 1.0)\n")
    
    print("--- Processing Full Batch Data ---")
    process_news()
