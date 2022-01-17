# The Emote-Me Journey

Welcome to Emote-Me! The vision for this project is to create a fun and interactive way for netizens to create their own custom cross-platform emotes. That sounds like a simple enough task, so I decided to spice things up a bit. Emote-Me will be built around a GAN, or Generative Adversial Network, which will randomly generate unique new characters for people to claim as their own. 

### It's about the journey, not the destination
At the end of the day, I'm just a curious learner trying to challenge myself in the world of AI and Machine Learning. I want this project to be less about the end result, and more about the lessons I learn along the way. I've broken down my journey into 4 main stages, which I will be exploring over the next few months and documenting along the way!

1. Dataset Generation and Processing
2. Model Training and Optimization (and re-training, and re-training, and re-training, ......)
3. Application UX and Deployment
4. Web3 and Beyond? 

## December 23rd 2021
The day it all began! I decided to begin generating my dataset using the [Google Images Scraper](https://github.com/ohyicong/Google-Image-Scraper). I played around with different keywords until I realized that I would actually fare better scraping [Etsy marketplace results](https://www.etsy.com/ca/search?q=chibi%20emote). You can check out my implementation [here](https://github.com/mariavmihu/emote-me/blob/main/webscrapers/EtsyScraper.py)!

The individual emote faces were labelled using [LabelImg](https://github.com/tzutalin/labelImg) and cropped using a custom python script until I had about 3,500 images to work with!

![alt text] (src/images/training_sample_set.png) <br>

## January 7th 2022
The first iteration of training and optimizing! I implemented a basic DCGAN similar to that in the official [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). The GAN started outputting complete and utter noise

![alt text] (https://github.com/mariavmihu/emote-me/blob/main/src/images/jan7_noise.jpg "noise sample")

![alt text] (https://github.com/mariavmihu/emote-me/blob/main/src/images/jan7_sample1.jpg "first sample")

![alt text] (https://github.com/mariavmihu/emote-me/blob/main/src/images/jan7_sample2.png "second sample")

The outputs are very noisy, but if you look deeper it's crazy how face-like the outputs are starting to look!

## January 10th 2022 - Present
I am currently working on playing around with some basic parameters and changes to the models before I move on to greater structural changes. The images are looking much less noisy, but now they are slightly less human-looking. More updates incoming as I polish things up ;) 

![alt text] (https://github.com/mariavmihu/emote-me/blob/main/src/images/jan10_sample1.jpg "first sample")
![alt text] (https://github.com/mariavmihu/emote-me/blob/main/src/images/jan10_sample2.jpg "second sample")
