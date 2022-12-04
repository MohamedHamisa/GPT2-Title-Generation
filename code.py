#scrape arxiv for titles

import arxivscraper as ax # to retrieve abstracts from given categories and date range
import numpy as np

'''
# scraper for arxiv stat.ml
scraper = ax.Scraper(category='stat', date_from='2017-08-01',
                     date_until='2019-07-01', t=10, 
                     filters={'categories':['stat.ml'],'abstract':['learning']})

# scraper for arxiv q-bio
scraper = ax.Scraper(category='q-bio', date_from='2016-08-01',
                     date_until='2019-07-01', t=10, 
                     filters={'categories':['q-bio.GN', 'q-bio.NC']})
'''

# scraper for arxiv physics
scraper = ax.Scraper(category='physics', date_from='2019-05-01',
                     date_until='2019-07-03', t=10,
                     filters={'categories':['quant-ph']})

output = scraper.scrape()



# cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
titles = [' '.join(o['title'].split()) for o in output]
np.savetxt('titles_ref.csv', np.array(titles), fmt='%s')


#alternatively, scrape something else

import urllib
import numpy as np

# scrape some interesting quotes
url = 'https://raw.githubusercontent.com/akhiltak/inspirational-quotes/master/Quotes.csv'
response = urllib.request.urlopen(url).read().decode()
quotes = []
lines = response.split('\n')
for line in lines[:-1]:
    quotes.append(line.split(';')[0].replace("\'", '').replace('*', '').replace('#', '').replace('%', '').replace('&', ''))
    
np.savetxt('titles.csv', np.array(quotes[1:]), fmt='%s')


#step 2 - finetune gpt2

import gpt_2_simple as gpt2

model_name = "117M" # "355M" for larger model (it's 1.4 GB)
gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              'titles_ref.csv',
              model_name=model_name,
              steps=1000,
              save_every=200,
              sample_every=25)   # steps is max number of training steps

gpt2.generate(sess)

# look at the model
sample_file = 'samples/samples-901'
t = open(sample_file, 'r').read()

for s in ['endoftext', 'startoftext', '<|', '|>']:
    t = t.replace(s, '')
for title in t.title().split('\n')[1:]:
    if not title == '':
        print('- ' + title)

#generating new samples from the finetuned model
generating new samples from the finetuned model
import gpt_2_simple as gpt2
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

#generate one sample
prefix = 'neural' # None is default
text = gpt2.generate(sess,
              length=40,
              temperature=0.7,
              prefix=neural,
              nsamples=1,
              batch_size=1,
              return_as_list=True
             )


t = text[0].title()
t = t.replace('<|Startoftext|>', '').replace('\n', '') # remove extraneous stuff
t = t[:t.index('<|Endoftext|>')] # only get one title
print(t)

#generate a bunch of samples
text = gpt2.generate(sess,
#               length=40,
              temperature=0.7,
              prefix=None,
              nsamples=100,
              batch_size=1,
              return_as_list=True
             )


t = text[0].title()
t = t.replace('<|Startoftext|>', '').replace('\n', '') # remove extraneous stuff
t = t[:t.index('<|Endoftext|>')] # only get one title
print(t)
