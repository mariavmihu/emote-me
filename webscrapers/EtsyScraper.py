"""
Created December 30th 2021.

Extension of GoogleImageScraper by @OHyic

@author: mariavmihu
"""
#import selenium drivers
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException       

#import helper libraries
import time
from PIL import Image

#import parent class
from GoogleImageScrapper import GoogleImageScraper

class EtsyScraper(GoogleImageScraper):
    
    def __init__(self,webdriver_path,image_path, search_key="cat",headless=False,min_resolution=(0,0),max_resolution=(1920,1080), page_limit=20):
        number_of_images = 64
        super().__init__(webdriver_path,image_path, search_key,number_of_images=number_of_images,headless=False,min_resolution=(0,0),max_resolution=(1920,1080))
        
        self.curr_page = 1
        self.url = self.format_query_string()
        self.page_limit = page_limit
        self.listing_urls = []

    def format_query_string(self):
        return "https://www.etsy.com/ca/search?q=%s&page=%s&ref=pagination"%(self.search_key, self.curr_page)

    def visit_search_pages(self):
        while self.curr_page <= self.page_limit:
            self.find_listing_urls()
            self.curr_page += 1
            self.url = self.format_query_string()

    def find_listing_urls(self):
        print("[INFO] Scraping for image link... Please wait.")
        
        count = 0
        missed_count = 0
        self.driver.get(self.url)
        time.sleep(5)
        
        # let's say N is the page number
        # FOR PAGES 1-4
        # the primary search results are always numbered 1-16 on a given page
        # the secondary search results are numbered from ((64*(n-1) - 16*(n-1)) + 1) to (64*n - 16*n)
        primary_index = 1
        if (self.curr_page < 5):
            primary_index_max = 16
        secondary_index = (64*(self.curr_page - 1) - 16*(self.curr_page - 1)) + 1
        secondary_index_max = 64*self.curr_page - 16*self.curr_page
        
        # FOR PAGE 5
        # the primary search results are always numbered 1-8 on a given page
        # the secondary search results are numbered from ((64*(n-1) - 16*(n-1)) + 1) to (64*n - 16*n)
        if (self.curr_page == 5):
            primary_index_max = 8
            self.number_of_images = 56
        
        # FOR PAGES 6-20
        # no primary search results
        # the secondary search results are numbered from ((64*(n-1) - 16*(n-1)) + 1) to (64*n - 16*n)
        if (self.curr_page > 5):
            primary_index_max = 0
            self.number_of_images = 48

        while self.number_of_images > count:
            print(count)
            print(self.number_of_images)
            #collect the urls of all the images on the page
            if primary_index <= primary_index_max:
                try:
                    listing_url = self.driver.find_element_by_xpath(f'//ul[@class="wt-grid wt-grid--block wt-pl-xs-0 tab-reorder-container"]/li/div/a[@data-position-num="{primary_index}"]').get_attribute('href')
                    
                    print(listing_url)
                    self.listing_urls.append(listing_url)
                    
                    time.sleep(1)
                    missed_count = 0 
                    primary_index += 1
                    count += 1
                except Exception as e:
                    print(e)
                    missed_count = missed_count + 1
                    if (missed_count>10):
                        print("[INFO] No more photos.")
                        break
                    
            if secondary_index <= secondary_index_max:
                try:
                    listing_url = self.driver.find_element_by_xpath(f'//div[@data-appears-component-name="search2_organic_listings_group"]/div/a[@data-position-num="{secondary_index}"]').get_attribute('href')
                    
                    print(listing_url)
                    self.listing_urls.append(listing_url)
                    
                    time.sleep(1)
                    missed_count = 0 
                    secondary_index += 1
                    count += 1
                except Exception as e:
                    print(e)
                    missed_count = missed_count + 1
                    if (missed_count>10):
                        print("[INFO] No more photos.")
                        break
                

        print("[INFO] Etsy Search Complete")
        return self.listing_urls



    def visit_listing_pages(self):
        
        image_urls = []

        
        for listing in self.listing_urls:
            self.driver.get(listing)
            time.sleep(3)
            
            carousel_position = 0
            
            '''            
            html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
            print (html)
            with open('dump.html', 'w', encoding='UTF-8') as dumper:
                dumper.write(html)'''
            
            try:
                carousel_exists = len(self.driver.find_elements_by_xpath(f'//button[@data-direction="prev"]'))
            except Exception as e:
                print(e)

            while carousel_position >= 0:
                
                try:
                    #might not be the most fool-proof way, but we will see
                    
                    main_image = self.driver.find_element_by_xpath('//ul[@class="wt-list-unstyled wt-overflow-hidden wt-position-relative carousel-pane-list"]/li/img').get_attribute('src') \
                        if carousel_position == 0 else \
                            self.driver.find_element_by_xpath('//li[@class="wt-position-absolute wt-width-full wt-height-full wt-position-top wt-position-left carousel-pane wt-animated--appear-01"]/img').get_attribute('src')
                    print(main_image)
                    image_urls.append(main_image)
                    
                except Exception as e:
                    print(e)
                
                carousel_exists = False
                
                if carousel_exists:
                    
                    try:
                        carousel_position += 1
                        
                        next_element = len(self.driver.find_elements_by_xpath(f'//div[@class="image-wrapper wt-position-relative carousel-container-responsive"]/div/div[2]/div[3]/ul/li[@data-index="{carousel_position}"]'))
                        if not next_element:
                            carousel_position = -1
                            break
                        
                        next_image = self.driver.find_element_by_xpath(f'//div[@class="image-wrapper wt-position-relative carousel-container-responsive"]/div/div[2]/div[3]/ul/li[@data-index="{carousel_position}"]')
                        print(next_image)
                        time.sleep(1)
                        
                        next_image.click()
                        time.sleep(2)
                    except Exception as e:
                        print(e)

                else:
                    carousel_position = -1
        
        return image_urls