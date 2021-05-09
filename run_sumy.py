from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pickle
import pandas as pd

LANGUAGE = "english"
SENTENCES_COUNT = 10


if __name__ == "__main__":
    #url = "https://en.wikipedia.org/wiki/Automatic_summarization"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    # parser = PlaintextParser.from_string("Check this out.", Tokenizer(LANGUAGE))
    
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    ted_file = open('./ted_data/processed_ted.pkl', 'rb')
    ted_main = pickle.load(ted_file)
    ted_file.close()
    #ted_main = ted_main[['description', 'transcript']]
    """
    summarized_teds = pd.DataFrame(columns=['summary'])
    for index, row in ted_main.iterrows():
        parser = PlaintextParser.from_string(row['expanded_transcript'], Tokenizer(LANGUAGE))
        summary = ' '.join([str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)])
        summarized_teds = summarized_teds.append({'summary': summary}, ignore_index=True)
    combined_df =  pd.concat([ted_main, summarized_teds], axis=1)
    print(combined_df.head(5))
    file_out = open('./ted_data/summarized_ted.pkl', 'wb')
    pickle.dump(summarized_teds, file_out)
    file_out.close()

    file_out = open('./ted_data/combined_ted.pkl', 'wb')
    pickle.dump(combined_df, file_out)
    file_out.close()
    #for sentence in summarizer(parser.document, SENTENCES_COUNT):
    #    print(sentence)
    """
    file_in = open('./ted_data/combined_ted.pkl', 'rb')
    summarized_ted = pickle.load(file_in)
    summarized_ted.to_csv('./ted_data/combined_ted.csv')
    file_in.close()
    print(summarized_ted.head(5))
