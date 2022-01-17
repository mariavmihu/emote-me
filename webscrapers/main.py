# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
from GoogleImageScrapper import GoogleImageScraper
from EtsyScraper import EtsyScraper
import os
import sys

if __name__ == "__main__":
    #Define file path
    webdriver_path = os.path.normpath(os.getcwd()+"\\webdriver\\chromedriver.exe")
    image_path = os.path.normpath(os.getcwd()+"\\photos")

    #Add new search key into array 
    search_keys_google_images= [
        'chibi girl',
        'chibi girl cute',
        'chibi girl small',
        'chibi girl face',
        'chibi girl simple',
        'chibi emote', 
        'chibi emote anime', 
        'chibi emote etsy', 
        'chibi twitch emote happy',
        'chibi twitch emote',
        'hyanna natsu chibi', #name of a Chibi artist
        'hyanna natsu emote'
        ]

    number_of_images = 1000
    
    
    headless = False
    min_resolution=(0,0)
    max_resolution=(9999,9999)

    '''
    #Main program
    for search_key in search_keys_google_images:
        image_scraper = GoogleImageScraper(webdriver_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
        image_urls = image_scraper.find_image_urls()
        image_scraper.save_images(image_urls)
    
    #Release resources    
    del image_scraper
    '''
    
    search_keys_etsy= [
        'chibi emotes'
    ]
    
    page_limit = 20
    
    for search_key in search_keys_etsy:
        image_scraper = EtsyScraper(webdriver_path,image_path,search_key,headless,min_resolution,max_resolution,page_limit)
        '''image_scraper.visit_search_pages()
        listing_urls = image_scraper.find_listing_urls()
        
        #save the listing urls in case something goes wrong
        print("saving listings to file, just in case")
        with open("DUMP_LISTINGS.txt", 'w') as f:
            f.write(str(listing_urls))
        f.close()
        print("done saving to listings file")'''
        
        image_urls = image_scraper.visit_listing_pages()
        image_scraper.save_images(image_urls)
        
        print(image_urls)
    
    del image_scraper