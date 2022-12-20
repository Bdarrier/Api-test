import spacy
from fastapi import FastAPI
from spacy.language import Language
from fonction import *
import pickle
import pandas as pd


#opening API
app = FastAPI()
@app.get("/titres/{titre}/questions/{question}/")
def get_tag_suggestion(titre,question):
    """
    definition fonction
    ___
    :parameter
    
    ___
    :return: 
    """
    # model loading
    model = pickle.load(open("PickledModel/model.pkl", 'rb'))
    binarized = pickle.load(open("PickledModel/binarizerr.pkl", 'rb'))
    vectorized = pickle.load(open("PickledModel/vectorizer.pkl", 'rb'))
    # prepare pipeline nlp
    Language.factory("language_detector", func=get_lang_detector)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector", last=True)
    #
    canned_soup=prepare_the_soup(question)
    titre_et_question='. '.join([str(titre) , str(canned_soup) ])
    tmp_text, tmp_lemma, tmp_sent_text, tmp_sent_lemma, langdict = lemmatisation(nlp(titre_et_question))
    tfidf_question=vectorized.transform([tmp_lemma])
    df_tfidf_question = pd.DataFrame(tfidf_question.toarray(), columns=vectorized.get_feature_names_out())
    suggested_tag_matrix = model.predict(df_tfidf_question)
    suggested_tag_words = binarized.inverse_transform(suggested_tag_matrix)
#
    return suggested_tag_words

