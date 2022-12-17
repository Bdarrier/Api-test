from bs4 import BeautifulSoup
from spacy_langdetect import LanguageDetector


def prepare_the_soup(give_the_soup):
    """
    definition
    ___
    :parameter

    ___
    :return: 
    """
    canned_soup = BeautifulSoup(give_the_soup, 'html.parser')
    soup_of_code = canned_soup.findAll('code')
    #
    for code in soup_of_code:
        code.replace_with(" ")
    #
    agg_soup = []
    for each in canned_soup:
        agg_soup.append(str(each.text))
    #
    hot_canned_soup = ' '.join(agg_soup)
    #
    return hot_canned_soup


def lemmatisation(doc_for_lemma):
    """
    definition
    ___
    :parameter

    ___
    :return:
    """
    return ([token.text for token in doc_for_lemma if (not token.is_stop | token.is_punct) and token.is_alpha],
            [token.lemma_ for token in doc_for_lemma if (not token.is_stop | token.is_punct) and token.is_alpha
             and (token.pos_ == 'NOUN')],
            [[' '.join(token.text for token in sent if (not token.is_stop | token.is_punct) and token.is_alpha)]
             for sent in doc_for_lemma.sents],
            [[' '.join(token.lemma_ for token in sent if ((not token.is_stop | token.is_punct) and token.is_alpha
                                                          and (token.pos_ == 'NOUN')))] for sent in
             doc_for_lemma.sents],
            doc_for_lemma._.language)


def get_lang_detector(nlp, name):
    """
    definition
    ___
    :parameter

    ___
    :return:
    """
    return LanguageDetector()
