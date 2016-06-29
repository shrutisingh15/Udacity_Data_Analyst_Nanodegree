#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()
    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        
        # project part 2: comment out the line below
        '''
        In parseOutText(), comment out the following line: 

        words = text_string 

        Augment parseOutText() so that the string it returns has all the words stemmed using a SnowballStemmer. Rerun parse_out_email_text.py, which will use your updated parseOutText() function--whats your output now?

        Hint: you'll need to break the string down into individual words, stem each word, then recombine all the words into one string.


        '''
        #words = text_string

        # split the text string into individual words, stem each word,
		
        stemmer = SnowballStemmer('english')
        input_string = text_string.split() #makes a list of words
        for i in range(len(input_string)):
            input_string[i] = stemmer.stem(input_string[i])
        # and append the stemmed word to words (make sure there's a single
        # space between each stemmed word)
            
        words = " ".join(input_string)# this -> " " ensures space b/w words        

    return words
    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

