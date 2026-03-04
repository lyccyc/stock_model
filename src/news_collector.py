import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import email.utils

def fetch_latest_0050_news():
    """
    Scrape the latest news headlines and snippets related to '0050 ETF', '元大台灣50', or '台股大盤走勢'
    from Google News RSS feed.
    """
    # The 'when:1d' query parameter restricts Google News to the last 24 hours.
    url = "https://news.google.com/rss/search?q=0050+ETF+OR+%E5%85%83%E5%A4%A7%E5%8F%B0%E7%81%A350+OR+%E5%8F%B0%E8%82%A1%E5%A4%A7%E7%9B%A4%E8%B5%B0%E5%8B%A2+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve news. Status code: {response.status_code}")
        return []

    # Using 'xml' parser since the feed is an RSS map 
    soup = BeautifulSoup(response.content, 'xml')
    items = soup.find_all('item')
    
    news_list = []
    
    # Calculate cutoff time for exactly 24 hours ago
    cutoff_time = datetime.now().astimezone() - timedelta(hours=24)
    
    for item in items:
        title = item.title.text if item.title else "No Title"
        description_ele = item.description
        
        # Google News RSS wraps snippet in HTML inside the description tag
        if description_ele:
            desc_soup = BeautifulSoup(description_ele.text, 'html.parser')
            snippet = desc_soup.get_text(strip=True)
        else:
            snippet = "No Snippet"
            
        pub_date_str = item.pubDate.text if item.pubDate else ""
        
        is_recent = True
        if pub_date_str:
            try:
                # Parse RFC 2822 date into aware datetime
                date_tuple = email.utils.parsedate_tz(pub_date_str)
                if date_tuple:
                    timestamp = email.utils.mktime_tz(date_tuple)
                    pub_date = datetime.fromtimestamp(timestamp).astimezone()
                    # Filter for only the past 24 hours
                    if pub_date < cutoff_time:
                        is_recent = False
            except Exception as e:
                pass
                
        if is_recent:
            news_list.append({
                'title': title,
                'snippet': snippet,
                'date': pub_date_str
            })
            
        if len(news_list) >= 5:
            break
            
    return news_list

def get_news_as_text():
    """
    Compiles the fetched news headlines and snippets into a single string formatting,
    which will be useful later for sentiment analysis.
    """
    news_items = fetch_latest_0050_news()
    if not news_items:
        return "No recent news found for 0050 ETF within the last 24 hours."
        
    combined_text = ""
    for idx, news in enumerate(news_items, 1):
        combined_text += f"[{idx}] Headline: {news['title']}\n"
        combined_text += f"    Snippet: {news['snippet']}\n"
        combined_text += f"    Published: {news['date']}\n\n"
        
    return combined_text.strip()

if __name__ == "__main__":
    print("Fetching the latest 0050 ETF news...\n")
    news_items = fetch_latest_0050_news()
    
    if news_items:
        import os
        import pandas as pd
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(news_items)
        output_filepath = os.path.join(output_dir, 'news_raw.csv')
        df.to_csv(output_filepath, index=False)
        print(f"Raw news successfully saved to {output_filepath}\n")
    
    news_text = get_news_as_text()
    if news_text:
        print("--- Latest 0050 ETF News Overview ---")
        print(news_text)
