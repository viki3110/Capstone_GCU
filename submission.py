#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Note for enterprising students:  There are myriad ways to split sentences for
    this algorithm.  For instance, you might want to exclude punctuation (unless
    it's organized in an email address format) or exclude numbers (unless they're
    organized in a zip code or phone number format).  Clearly this can become quite
    complex.  For our purposes, please split using the space character ONLY.  This
    is intended to balance your understanding with our ability to autograde the
    assignment.  Thanks and have fun with the rest of the assignment!

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split(" ")
    # *** END CODE HERE ***

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least *five messages*.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    add=0
    dict_of_word={} # Defining the word dictionary
    for i in messages: # loop to go through each message
        list_of_word= get_words(i)
        for j in set(list_of_word):# loop to go through each words
            if j in set(dict_of_word):
                dict_of_word[j] +=1
            else:
                dict_of_word[j]=1
                
    #We need to remove the rare word which appears less than 5 times
    add=0
    dict_of_word={} # Defining the word dictionary
    for i in messages: # loop to go through each message
        list_of_word= get_words(i)
        for j in set(list_of_word):# loop to go through each words
            if j in set(dict_of_word):
                dict_of_word[j] +=1
            else:
                dict_of_word[j]=1
                
    #We need to remove the rare word which appears less than 5 times
    k=0 #initializing indexes
    new_dict={}
    for key,value in list(dict_of_word.items()): #this loop will check if rare word appearing less than 5 time
        if value >=5:
            new_dict[key] = k
            k+=1
     
    return new_dict

   # k=0 #initializing indexes
   # for key in set(dict_of_word.keys()): #this loop will check if rare word appearing less than 5 time
   #     if dict_of_word[key] >=5:
    #        dict_of_word[key]=add
     #       add += 1
     #   else:
     #       del dict_of_word[key]
   # return dict_of_word
 
   
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to *a word of the vocabulary*.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    
    num=len(word_dictionary)# length of word dictionary
    arr=np.array([]).reshape(0,num) #creating array and reshaping it
    for i in messages:
        list_of_word=get_words(i)
        count_of_words=np.zeros((1,num))
        for word in list_of_word:
            if word in word_dictionary:
                count_of_words[0,word_dictionary[word]] +=1
        arr=np.vstack([arr,count_of_words])
    return arr
    
    
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    x,y = matrix.shape
    mat_y0=matrix[labels==0, :].sum(axis=0)# this would create all y's with label 0
    mat_y1=matrix[labels==1, :].sum(axis=0)## this would create all y's with label 1
    param2=(mat_y0+1)/(mat_y0.sum()+y)
    param1=(mat_y1+1)/(mat_y1.sum()+y)
    param3=np.mean(labels)
    return(param1,param2,param3) # this would return 3 parameter
    
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    param1,param2,param3=model # unpacking 3 paramemeter from the fit function
    lgp1=(np.log(param1)*matrix).sum(axis=1)+np.log(param3)
    lgp0=(np.log(param2)*matrix).sum(axis=1)+np.log(1-param3)
    return (lgp1>lgp0).astype(np.int)
   
    
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    
    param1,param2,param3=model #Unpacking parameter
    ind=(-np.log(param1/param2)).argsort()[:5]

    dct=[w for w in dictionary.keys()]
    #for w in dictionary.keys():
    #    W=w#dct=[w]
    #dct=W
    
    return [dct[i] for i in ind]
    
    
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accr = 0
    for rad in radius_to_consider:
        pred = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, rad)
        accr1 = np.mean(pred == val_labels)
        if accr1 > accr:
            accr = accr1
            opt_rad = rad
    return opt_rad        
    
    # *** END CODE HERE ***


def main():
    #train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    train_messages, train_labels = util.load_spam_dataset('D:\Stanford AI\Git\XCS229i-PS4\XCS229i-PS4-master\src-spam\spam_train.tsv')
   # D:\Stanford AI\Git\XCS229i-PS4\XCS229i-PS4-master\src-spam
    val_messages, val_labels = util.load_spam_dataset('D:\Stanford AI\Git\XCS229i-PS4\XCS229i-PS4-master\src-spam\spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('D:\Stanford AI\Git\XCS229i-PS4\XCS229i-PS4-master\src-spam\spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary_(soln)', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix_(soln)', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions_(soln)', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words_(soln)', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius_(soln)', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()


# In[ ]:




