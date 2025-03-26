import ollama
import google.generativeai as genai
from google_play_scraper import reviews, Sort

def get_google_play_reviews(app, count=100, filter_score_with=None,
                            lang="en", country="us", sort=Sort.NEWEST):
    """
    Wrapper of google_play_scrapper
    app: the code of the app you want to scan (eg: 'com.binance.dev')
    count: the number of reviews (defaults to 100)
    filter_score_with: if you want to filter reviews by number of stars
    """
    result, continuation_token = reviews(
            app,
            count=count, # defaults to 100
            filter_score_with=filter_score_with, # defaults to None(means all score)
            lang=lang, # defaults to 'en'
            country=country, # defaults to 'us'
            sort=sort # defaults to Sort.NEWEST
        )
    return result


def gemini_query(prompt, gemini_key, categories="", input_text="", counter=0):
    prompt = prompt
    categories = categories
    input_text = input_text
    
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    try:
        response = model.generate_content(prompt.format(input_text=input_text,categories=categories))
    except: 
        time.sleep(5)
        response = "gemini failed to respond"
        return response
    return response.text.strip()

def ollama_query(prompt, model = "llama3.2:1b", categories="", input_text=""):
    #model = "deepseek-r1:7b"
    prompt = prompt
    
    response = ollama.chat(
        model = model,
        messages = [{"role":"user", "content":prompt.format(input_text=input_text,categories=categories )}]    
    )

    return response["message"]["content"].strip()