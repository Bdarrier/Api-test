from bs4 import BeautifulSoup


def prepare_the_soup(give_the_soup):
    """
    Brew soup in can : parsing htm code from StackOverflow.
    Implemented for dealing with future input
    ___
    str(string)
    ___
    parsed string
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
    get lemmatised noun only token from text.
    ___
    str to parse
    ___
    lemmatised noun token
    """
    return ([token.lemma_ for token in doc_for_lemma if (not token.is_stop | token.is_punct) and token.is_alpha
             and (token.pos_ == 'NOUN')])
