
import re, nltk, pickle, argparse
import os
import data_helper
from features import get_features_category_tuples

DATA_DIR = "data"


def write_features_category(features_category_tuples, output_file_name):
    output_file = open("{0}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()

def write_all_results(confusion_matrix, accuracy, feature_set, eval_file, clearflag=True):
    if clearflag == True:
        output_file = open("all-results.txt", "w", encoding="utf-8")
    else:
        output_file = open("all-results.txt", "a", encoding="utf-8")
        output_file.write("Testing: {0}\nThe accuracy of {1} is: {2}\nConfusion Matrix: \n\n{3}\n".format(eval_file, feature_set, accuracy, str(confusion_matrix)))
    output_file.close()




def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):

    ###     YOUR CODE GOES HERE
    # TODO: evaluate your model
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    features_only = [target[0] for target in features_category_tuples]
    reference_labels = [target[1] for target in features_category_tuples]
    predicted_labels = classifier.classify_many(features_only)
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)



    return accuracy, confusion_matrix

def eval_test(classifier, features_category_tuples):
    features_only = [target[0] for target in features_category_tuples]
    predicted_labels = classifier.classify_many(features_only)
    output_file = open("restaurant-competition-model-P1-predictions.txt", "w", encoding="utf-8")
    output_file.write(str(predicted_labels) + "\n")
    output_file.close()



def build_features(data_file, feat_name, save_feats=None, binning=False):
    # read text data
    if "tsv" in data_file:
        positive_texts, negative_texts = data_helper.get_reviews(os.path.join(DATA_DIR, data_file))
        category_texts = {"positive": positive_texts, "negative": negative_texts}
    else:
        text = data_helper.get_reviews(os.path.join(DATA_DIR,data_file))
        category_texts = {"": text}


    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)

    return features_category_tuples, texts



def train_model(datafile, feature_set, save_model=None):

    features_data, texts = build_features(datafile, feature_set)

    ###     YOUR CODE GOES HERE
    # TODO: train your model here
    write_features_category(features_data, feature_set + "-training")
    classifier = nltk.classify.NaiveBayesClassifier.train(features_data)


    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier

#file, feature set, eval file
def train_eval(train_file, feature_set, eval_file=None):

    # train the model
    split_name = "train"
    model = train_model(train_file, feature_set, eval_file)
    model.show_most_informative_features(20)
    # for feature in model.most_informative_features(20):
    #     



    # save the model
    if model is None:
        model = get_classifier(classifier_fname)

    # evaluate the model
    if eval_file is not None:
        features_data, texts = build_features(eval_file, feature_set, save_feats=None)
        if 'test.txt' in eval_file:
            eval_test(model, features_data)
        accuracy, cm = evaluate(model, features_data, texts, data_set_name=None)
        save_classifier(model, feature_set + "-development")
        write_features_category(features_data, (feature_set + "-development") )
        write_all_results(cm, accuracy, feature_set, eval_file, clearflag=False)
        if feature_set == "word_features" and eval_file == "dev_examples.tsv":
            output_file = open("output-ngrams.txt", "w", encoding="utf-8")
            output_file.write("The accuracy of {} is: {}\n".format(eval_file, accuracy))
            output_file.write("Confusion Matrix:\n")
            output_file.write(str(cm))
            output_file.close()
        if feature_set == "word_pos_features" and eval_file == "dev_examples.tsv":
            output_file = open("output-pos.txt", "w", encoding="utf-8")
            output_file.write("The accuracy of {} is: {}\n".format(eval_file, accuracy))
            output_file.write("Confusion Matrix:\n")
            output_file.write(str(cm))
            output_file.close()
        if feature_set == "word_pos_liwc_features" and eval_file == "dev_examples.tsv":
            output_file = open("output-liwc.txt", "w", encoding="utf-8")
            output_file.write("The accuracy of {} is: {}\n".format(eval_file, accuracy))
            output_file.write("Confusion Matrix:\n")
            output_file.write(str(cm))
            output_file.close()
        print("The accuracy of {} is: {}".format(eval_file, accuracy))
        print("Confusion Matrix:")
        print(str(cm))
    else:
        accuracy = None

    return accuracy


def main():


    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('-d', dest="data_fname", default="train_examples.tsv",
                        help='File name of the testing data.')
    args = parser.parse_args()


    train_data = args.data_fname


    eval_data = "dev_examples.tsv"

    #clear all-results.txt
    write_all_results("", "", "", "", clearflag=True)

    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features"]:
        print("\nTraining with {}".format(feat_set))
        acc = train_eval(train_data, feat_set, eval_file=eval_data)
        acc2= train_eval(train_data, feat_set, eval_file="train_examples.tsv")
        wut = train_eval(train_data, feat_set, eval_file="test.txt")


    output_file = open("all-results.txt", "a", encoding="utf-8")
    output_file.write("\nThe feature set which resulted in the best accuracy was the word_features. However, when testing with the liwc features, I found that the accuracies increase when you have a selected amount of liwc features, rather than using all of them. This is due to there being a vast amount of data, which can lead to more instances of misclassification. Therefore, I selected a handful of liwc features which brought 76%~ accuracy and is commented out in the features file. I found that classifying the negative texts was less fruitful than the positive texts, so I chose features that were the baselines for what would fuel myself to write a negative review.")
    output_file.close()

    


if __name__ == "__main__":
    main()




