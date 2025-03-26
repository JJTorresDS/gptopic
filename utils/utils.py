import ollama
import google.generativeai as genai
from google_play_scraper import reviews, Sort
import time

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


def gemini_query(prompt, gemini_key,debug=False, counter=0, tries=3):
    
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    if counter < tries:
        try:
            response = model.generate_content(prompt)
        except: 
            
            if debug:
                print("Gemini Failed to respond. Sleeping...")
            time.sleep(10) 

            if debug:
                print("Entering recursive step.", counter+1)
            return gemini_query(prompt=prompt, 
                                gemini_key=gemini_key,
                                  counter=counter+1)
    else:
        return "gemini failed to respond"
    return response.text.strip()

def ollama_query(prompt, model = "llama3.2:1b"):
    #model = "deepseek-r1:7b"
    
    response = ollama.chat(
        model = model,
        messages = [{"role":"user", "content":prompt }]    
    )

    return response["message"]["content"].strip()