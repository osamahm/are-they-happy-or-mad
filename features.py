
import nltk
import re
import word_category_counter
import data_helper
import os, sys

DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)


def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token

    else:

        ###     YOUR CODE GOES HERE 

        #set stopwords
        stopwords = nltk.corpus.stopwords.words('english')

        #if lowercased token is not in stop words and is a word character
        if token.lower() not in stopwords and re.search(r'\w', token):

        	#lowercase and return
            normalized_token = token.lower()
        else:
            normalized_token = None

    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence

    ###     YOUR CODE GOES HERE

    #tokenize sentences
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
    	#grab word and pos tag
        for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        	#normalize the word
            normalized = normalize(word)

            #if the word is in our set of normalized words append
            if normalized:
                words.append(normalized)
                tags.append(tag)
    return words, tags

def get_ngram_features(tokens, binning=None):

	feature_vectors = {}

	uni_dist = nltk.FreqDist(nltk.ngrams(tokens,1))
	bi_dist = nltk.FreqDist(nltk.ngrams(tokens, 2))

	if binning:
		for word,count in uni_dist.items():
			feature_vectors["UNI_{0}".format(word[0])] = bin(count)
		for word,count in bi_dist.items():
			feature_vectors["BIGRAM_{0}_{1}".format(word[0],word[1])] = bin(count)
	else:
		for word,count in uni_dist.items():
			feature_vectors["UNI_{0}".format(word[0])] = round(uni_dist.freq(word),3)
		for word,count in bi_dist.items():
			feature_vectors["BIGRAM_{0}_{1}".format(word[0],word[1])] = round(bi_dist.freq(word),3)

	return feature_vectors


def get_pos_features(tags, binning=False):

	feature_vectors = {}

	uni_dist = nltk.FreqDist(nltk.ngrams(tags,1))
	bi_dist = nltk.FreqDist(nltk.ngrams(tags,2))

	if binning:
		for word,count in uni_dist.items():
			feature_vectors["UNIPOS_{0}".format(word[0])] = bin(count)
		for word,count in bi_dist.items():
			feature_vectors["BIGRAMPOS_{0}_{1}".format(word[0],word[1])] = bin(count)

	else:
		for word,count in uni_dist.items():
			feature_vectors["UNIPOS_{0}".format(word[0])] = round(uni_dist.freq(word),3)
		for word,count in bi_dist.items():
			feature_vectors["BIGRAMPOS_{0}_{1}".format(word[0],word[1])] = round(bi_dist.freq(word),3)

	return feature_vectors








def bin(count):
    """
    Results in bins of  0, 1, 2, 3 >=
    :param count: [int] the bin label
    :return:
    """
    the_bin = None
    ###     YOUR CODE GOES HERE

    the_bin = count if count < 5 else 6

    return the_bin


def get_liwc_features(words, binning=None):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    # TODO: binning

    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)
    liwc_keys = liwc_scores.keys()
    # print(liwc_keys)
    for target in liwc_keys:
        feature_vectors["liwc:{0}".format(target)] = bin(liwc_scores[target])



    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    # negative_score = liwc_scores["Negative Emotion"]
    # positive_score = liwc_scores["Positive Emotion"]
    # fps_score = liwc_scores["First Person Singular"]
    # affective_process_score = liwc_scores["Affective Processes"]
    # perceptual_score = liwc_scores["Perceptual Processes"]
    # cognitive_score = liwc_scores["Cognitive Processes"]    
    # anxiety_score = liwc_scores["Anxiety"]
    # anger_score = liwc_scores["Anger"]
    # health_score = liwc_scores["Health"]
    # leisure_score = liwc_scores["Leisure"]
    # time_score = liwc_scores["Time"]
    # certainty_score = liwc_scores["Certainty"]
    # discrepency_score = liwc_scores["Discrepency"]
    # communication_score = liwc_scores["Communication"]
    # inclusive_score = liwc_scores["Inclusive"]


    # feature_vectors["Negative Emotion"] = bin(negative_score)
    # feature_vectors["Positive Emotion"] = bin(positive_score)
    # feature_vectors["First Person Singular"] = bin(fps_score)
    # feature_vectors["Affective Processes"] = bin(affective_process_score)
    # feature_vectors["Anxiety"] = bin(anxiety_score)
    # feature_vectors["Anger"] = bin(anger_score)
    # feature_vectors["Time"] = bin(time_score)



    # if positive_score > negative_score:
    #     feature_vectors["liwc:positive"] = 1
    # else:
    #     feature_vectors["liwc:negative"] = 1

    return feature_vectors

def get_data(data_file):
	if "tsv" in data_file:
		positive_texts, negative_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))
		return positive_texts, negative_texts
	else:
		category_text = data_helper.get_reviews(os.path.join(DATA_DIR,data_file))
	return category_text


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features"}

def get_features_category_tuples(category_text_dict, feature_set, binning=None):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}

            ###     YOUR CODE GOES HERE
            if feature_set == "word_features":
                feature_vectors = get_ngram_features(words, binning=True)
            if feature_set == "word_pos_features":
                feature_vectors = get_ngram_features(words, binning=True)
                feature_vectors.update(get_pos_features(tags, binning=True))
            if feature_set == "word_pos_liwc_features":
                feature_vectors = get_ngram_features(words, binning=True)
                feature_vectors.update(get_pos_features(tags, binning=True))
                feature_vectors.update(get_liwc_features(words, binning=True))
            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts



def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write("{0:<10s}\t{1}\n".format(category, features))

def write_features(categoryType, reviewArray, outfile_name):
    with open(outfile_name,"w", encoding="utf-8") as fout:
        for review in reviewArray:
            if categoryType:
                fout.write(categoryType + " " + str(review) + "\n")
            else:
                fout.write(str(review) + "\n")
    fout.close()

def append_features(categoryType, reviewArray, outfile_name):
    with open(outfile_name,"a", encoding="utf-8") as fout:
        for review in reviewArray:
            if categoryType:
                fout.write(categoryType + " " + str(review) + "\n")
            else:
                fout.write(str(review) + "\n")
    fout.close()




def populate(targets):
	wordArray    = []
	tagArray     = []
	ngramArray   = []
	posArray     = []
	liwcArray    = []

	for target in targets:
		words, tags = get_words_tags(target)
		wordArray.append(words)
		tagArray.append(tags)
		ngramArray.append(get_ngram_features(words, binning=True))
		posArray.append(get_pos_features(tags, binning=True))
		liwcArray.append(get_liwc_features(words, binning=True))
	return wordArray, tagArray, ngramArray, posArray, liwcArray



def train():
    data_file = "train_examples.tsv"
    pos_data,neg_data = get_data(data_file)
    pos_wordArray, pos_tagArray, pos_ngramArray, pos_posArray, pos_liwcArray = populate(pos_data)
    neg_wordArray, neg_tagArray, neg_ngramArray, neg_posArray, neg_liwcArray = populate(neg_data)

    #debug()

    positive = "positive:" 
    negative = "negative:"

    write_features(positive, pos_ngramArray, "word_features-training-features.txt")
    append_features(negative, neg_ngramArray, "word_features-training-features.txt")

    write_features(positive, pos_ngramArray, "word_pos_features-training-features.txt")
    append_features(negative, neg_ngramArray, "word_pos_features-training-features.txt")
    append_features(positive, pos_posArray, "word_pos_features-training-features.txt")
    append_features(negative, neg_posArray, "word_pos_features-training-features.txt")

    write_features(positive, pos_ngramArray, "word_pos_liwc_features-training-features.txt")
    append_features(positive, pos_posArray, "word_pos_liwc_features-training-features.txt")
    append_features(positive, pos_liwcArray, "word_pos_liwc_features-training-features.txt")
    append_features(negative, neg_ngramArray, "word_pos_liwc_features-training-features.txt")
    append_features(negative, neg_posArray, "word_pos_liwc_features-training-features.txt")    
    append_features(negative, neg_liwcArray, "word_pos_liwc_features-training-features.txt")

def dev():
    data_file = "dev_examples.tsv"
    pos_data, neg_data = get_data(data_file)
    pos_wordArray, pos_tagArray, pos_ngramArray, pos_posArray, pos_liwcArray = populate(pos_data)
    neg_wordArray, neg_tagArray, neg_ngramArray, neg_posArray, neg_liwcArray = populate(neg_data)

    #debug()

    positive = "positive:" 
    negative = "negative:"

    write_features(positive, pos_ngramArray, "word_features-development-features.txt")
    append_features(negative, neg_ngramArray, "word_features-development-features.txt")

    write_features(positive, pos_ngramArray, "word_pos_features-development-features.txt")
    append_features(negative, neg_ngramArray, "word_pos_features-development-features.txt")
    append_features(positive, pos_posArray, "word_pos_features-development-features.txt")
    append_features(negative, neg_posArray, "word_pos_features-development-features.txt")

    write_features(positive, pos_ngramArray, "word_pos_liwc_features-development-features.txt")
    append_features(positive, pos_posArray, "word_pos_liwc_features-development-features.txt")
    append_features(positive, pos_liwcArray, "word_pos_liwc_features-development-features.txt")
    append_features(negative, neg_ngramArray, "word_pos_liwc_features-development-features.txt")
    append_features(negative, neg_posArray, "word_pos_liwc_features-development-features.txt")    
    append_features(negative, neg_liwcArray, "word_pos_liwc_features-development-features.txt")

def test():
    data_file = "test.txt"
    data = get_data(data_file)
    wordArray, tagArray, ngramArray, posArray, liwcArray = populate(data)

    write_features(None, ngramArray, "word_features-testing-features.txt")
    write_features(None, ngramArray, "word_pos_features-testing-features.txt")
    append_features(None, posArray, "word_pos_features-testing-features.txt")
    write_features( None , ngramArray, "word_pos_liwc_features-testing-features.txt")
    append_features( None, posArray, "word_pos_liwc_features-testing-features.txt")
    append_features( None , liwcArray, "word_pos_liwc_features-testing-features.txt")



if __name__ == "__main__":
    # train()
    # dev()
    # test()
    pass 

    

