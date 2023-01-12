import spacy
from fastapi import FastAPI
from fonction import *
import pickle
import pandas as pd

# model loading
logit = pickle.load(open("PickledModel/LogitReg.pkl", 'rb'))
lda = pickle.load(open("PickledModel/LDA_opti.pkl", 'rb'))
binarized = pickle.load(open("PickledModel/binarizer.pkl", 'rb'))
vectorized = pickle.load(open("PickledModel/vectorizer.pkl", 'rb'))
# nlp pipe

try:
    nlp = spacy.load("en_core_web_sm")
except:  # If not present, we download
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# opening API
app = FastAPI()


@app.get("/titres/{titre}/questions/{question}/")
def get_tag_suggestion(titre, question):
    """
    main fonction for tag suggestion and loading model
    ___
    Use title and question.
    Model where pretrained on StackOverflow data
    ___
    Tags suggestions
    """

    # Transformation in features
    canned_soup = prepare_the_soup(question)
    titre_et_question = '. '.join([str(titre), str(canned_soup)])
    tmp_lemma = lemmatisation(nlp(titre_et_question))
    tfidf_question = vectorized.transform([tmp_lemma])
    df_tfidf_question = pd.DataFrame(tfidf_question.toarray(), columns=vectorized.get_feature_names_out())

    # Unsupervised model LDA
    tfidf_lda = lda.transform(df_tfidf_question)
    suggested_tag_words_unsupervised = [vectorized.get_feature_names_out()[i] for i in
                                        lda.components_[tfidf_lda.argmax()].argsort()[:-5 - 1:-1]]

    # Supervised model logit
    #suggested_tag_matrix = logit.predict(df_tfidf_question)
    #suggested_tag_words_supervised = binarized.inverse_transform(suggested_tag_matrix)

    return {"LDA": suggested_tag_words_unsupervised, "LOGIT": "not_supported_by_heroku" }
