#!/bin/python


def read_files(tarfname, vocab = None):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    
    # Open Tar file
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    # Get file names for dev and test data
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
    #create empty class for storing data 
    class Data: pass
    sentiment = Data()

    #extract train data from file
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    #extract dev data from file
    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")

    #Create Tokenizer
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    def tokenizer(text):
        #create tokens
        tkns = nltk.tokenize.word_tokenize(text)
        #lematize tokens
        lemmas = list(map(nltk.stem.WordNetLemmatizer().lemmatize, tkns))
        return lemmas

    ## create feature matrix using TfidfVectorizer 
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenizer, vocabulary=vocab)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    ## Encode labels for model 
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    
    #open tarfile
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    #create empty class for storing data 
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    #get file name for unlabeled data
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
    #extract unlabled data from file
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip().lower()
        unlabeled.data.append(text)
        
    ## create feature matrix using TfidfVectorizer 
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    #extract data and labels from file in tar
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text.lower())
    return data, labels


if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    #get data from files
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    #train model using training data
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    #evaluate accuracy of model on training data
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    #evaluate accuracy of model on dev data
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

## find best features to use for feature selection
def find_best_features():
    import numpy as np
    import classify
    #read files
    sentiment = read_files("data/sentiment.tar.gz")
    #train model normally
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    #sort words by most decisive
    coef = cls.coef_[0]
    max = 100
    sortedIndices = np.argsort(coef)[-max:]
    top_max_words = []
    for i in sortedIndices:
        top_max_words.append(sentiment.count_vect.get_feature_names_out()[i])

    #set up variables to track results
    class results:pass
    results.k_list = []
    results.k_accuracy = []
    results.max_k = -1
    results.max_accuracy = -1
    # iterate over lists of k most decive and run and test model with those features
    for k in range(1,max):
        top_k_words = top_max_words[-k:]
        sentiment_k = read_files("data/sentiment.tar.gz",top_k_words)
        cls_k = classify.train_classifier(sentiment_k.trainX, sentiment_k.trainy)
        acc_k = classify.evaluate(sentiment_k.devX, sentiment_k.devy, cls_k, 'dev', willPrint=False)
        #save accuracy scores
        results.k_list.append(k)
        results.k_accuracy.append(acc_k)
        #determine max
        if acc_k > results.max_accuracy:
            results.max_k = k
            results.max_accuracy = acc_k
    #print max
    print("MAX K: ",results.max_k)
    print("MAX ACCURACY: ",results.max_accuracy)
    #plot accuracy by k
    import matplotlib.pyplot as plt
    plt.plot(results.k_list,results.k_accuracy)
    plt.xlabel('Top X Features used')
    plt.ylabel('Accuracy of Model')
    plt.title('Accuracy of Model by Number of Top Features Used')
    plt.show()

        
# find_best_features()