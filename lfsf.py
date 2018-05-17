import bioc
import os
import nltk
from Bio import pairwise2
import _pickle
import string
import jellyfish
from sklearn import tree
from operator import itemgetter
import numpy as np

from sklearn.model_selection import cross_val_score

#load BioC dataset
def load(dataset):
    src = os.path.join('data', dataset)
    with open(src) as fp:
        collection = bioc.load(fp)
    return collection

def is_abbr(word,abbr):

    if sum(1 for c in word if c.isupper()) > 2 and len(word) < 11:# and '-' not in word:
        return True

    if word in abbr:
        return True
    else:
        res = False
        iter = 0
        tmp = word.split('-')
        for t in tmp:
            #tx= ''.join(e for e in t if e.isalnum())
            if t in abbr:
                iter += 1
                break

        if iter == len(tmp):
            res = True


        return res

#extract raw text from BioC
def get_text(collection):
    try:
        i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]
        return [x.passages[i].text for x in collection.documents if x.passages[i].text != 'No abstract']
    except:
        return -1

#get SF and LF from BioC
def abbr_short_long(collection): #from abstract only w/o title

    short= []
    long = []
    #find index of abstract
    i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]

    #get addnotations (SF and LF) from abstracts
    try:
        annotations = [x.passages[i].annotations for x in collection.documents if len(x.passages[i].annotations) >0]
    except:
        return None

    #for each addnotations extract (SF and LF)
    for anno in annotations:
        short.extend([x.text for x in anno if 'SF' in x.id])
        long.extend([x.text for x in anno if 'LF' in x.id])


    return [(short[i],long[i]) for i,x in enumerate(short)]

#get SF from BioC
def abbr_short(collection): #from abstract only w/o title
    short= []
    long = []
    i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]

    try:
        annotations = [x.passages[i].annotations for x in collection.documents if len(x.passages[i].annotations) >0]
    except:
        return None

    for anno in annotations:
        short.extend([x.text for x in anno if 'SF' in x.id])
    return short

#find SF and its location
def abbr_short_long_location_sentences(collection):  # from abstract only w/o title

    i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]

    try:
        annotations = [x.passages[i].annotations for x in collection.documents if len(x.passages[i].annotations) > 0]
        text = [x.passages[i].text for x in collection.documents if len(x.passages[i].text) > 0 and x.passages[i].text != 'No abstract']
    except:
        return -1


    short = []
    long = []
    #location_short = []
    #location_long = []
    for anno in annotations:
        short.extend([(x.text,x.locations[0]) for x in anno if 'SF' in x.id])
        long.extend([(x.text,x.locations[0]) for x in anno if 'LF' in x.id])

    clear = [ (x[0],long[i][0]) for i,x in enumerate(short) if x[1].offset > long[i][1].offset and x[1].offset - long[i][1].end <3]
    #clear = [ (x[0],long[i][0]) for i,x in enumerate(short) ]
    # for anno in annotations:
    #     short.extend([x.text for x in anno if 'SF' in x.id])
    #     long.extend([x.text for x in anno if 'LF' in x.id])

    sentences = []
    for abs in text:
        sent = nltk.sent_tokenize(abs)

        for i in range(len(clear)):
            for s in sent:
                if clear[i][0] in s and clear[i][1] in s:
                    sentences.append((clear[i][0],clear[i][1],s))


    #sentences = [x for x in sentences if len(x[0]) < len(x[1]) and len(x[0]) <11 and any(s.isupper() for s in x[0]) and all(s.isalnum() for s in x[0]) and len(x[0])>1]
    sentences = list(set(sentences))

    return sentences

#remove duplicates + assumptions on SF/LF
def clean_short_long(short_long):
    short_long = list(set(short_long))
    short_long = [x for x in short_long if len(x[1])>len(x[0])]
    short_long = [x for x in short_long if len(x[0]) > 1]
    #short_long = [x for x in short_long if len(x[1].split()) > 1]
    short_long = [x for x in short_long if len(x[0].split()) == 1]
    #short_long = [x for x in short_long if len([y for y in x[0] if y.isupper()]) > 0]

    return short_long

# Transformation for SF detection; downstream
def transformation_list(collection):

    def mod_word(word):
        new = []
        for c in word:
            if c.isupper():
                new.append('U')
            elif c.islower():
                new.append('l')
            elif c.isdigit():
                new.append('d')
            else:
                new.append(c)
        return ''.join(new)

    i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]

    try:
        textandids = [[x.id,x.passages[i].text] for x in collection.documents if x.passages[i].text  != 'No abstract']
    except:
        return -1

    abbr = abbr_short(collection)
    #abbr_an = [''.join(e for e in x if e.isalnum()) for x in abbr]
    final = []
    for abstract in textandids:
        words = abstract[1].split()
        #words = list(set(words_d)) ##########DUPLICATES
        tmp = []
        for i,word in enumerate(words):
            tmp.append([word,words[i-1][-1:] +' ' + word,mod_word(words[i-1][-1:] +' ' + word),is_abbr(word,abbr)])

        final.extend([[abstract[0]] + x for x in tmp])

    return final

# Transformation for SF detection; downstream and upstream
def transformation_list_bothside(collection):

    def mod_word(word):
        new = []
        for c in word:
            if c.isupper():
                new.append('U')
            elif c.islower():
                new.append('l')
            elif c.isdigit():
                new.append('d')
            else:
                new.append(c)
        return ''.join(new)

    i = [i for i, x in enumerate(collection.documents[0].passages) if x.infons['type'] == 'abstract'][0]

    try:
        textandids = [[x.id, x.passages[i].text] for x in collection.documents if x.passages[i].text != 'No abstract']
    except:
        return -1

    abbr = abbr_short(collection)
    #abbr_an = [''.join(e for e in x if e.isalnum()) for x in abbr]
    final = []
    for abstract in textandids:
        words = abstract[1].split()
        #words = list(set(words_d)) ##########DUPLICATES
        tmp = []
        for i,word in enumerate(words):
            try:
                tmp.append([word,words[i-1][-1:] +' ' + word +' ' + words[i+1][:1]  ,mod_word(words[i-1][-1:] +' ' + word +' ' + words[i+1][:1] ),is_abbr(word,abbr)])
            except:
                tmp.append([word, words[i - 1][-1:] + ' ' + word + ' ' + words[0][:1],
                            mod_word(words[i - 1][-1:] + ' ' + word + ' ' + words[0][:1]), is_abbr(word, abbr)])

        final.extend([[abstract[0]] + x for x in tmp])

    return final

# unique elements by key
def unique_by_key(elements, key=None):
    if key is None:
        # no key: the whole element must be unique
        key = lambda e: e
    return list({key(el): el for el in elements}.values())

#test acronym detection - DT classyfier
def decition_tree_accronym(datasets):

    def tree_tab(tab):
        out =[]
        for a in tab:
            new = []
            new.append(a[3].count('U'))
            new.append(a[3].count('l'))
            new.append(a[3].count('d'))
            new.append(len(a[3]) - sum(new))
            new.append(a[4])

            out.append(new)

        return out

    trans = []
    trans_2 = []
    abbr = []


    for dataset in datasets:
        data = load(dataset)
        abbr.extend(abbr_short(data))  # list of acronyms
        trans.extend(transformation_list(data))

        trans_uniq = unique_by_key(trans, key=itemgetter(3))

    tab = tree_tab(trans_uniq)

    X = np.array([x[:-1] for x in tab])
    Y= np.array([x[-1:] for x in tab])
    ids = np.arange(len(X))
    np.random.shuffle(ids)
    X = X[ids]
    Y = Y[ids]
    features = ["id", "word", "extended word", "transformed", "label"]

    clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(X,Y)
    scores = cross_val_score(clf, X, Y, cv = 5, scoring = 'f1')
    test = 0

    #print(scores)
    print("Decision Trees")
    print("Average F1 score: " + str(sum(scores)/5))

#simple algorithm for LF detection using rules based algorithms
def long_form_detection(datasets):

    def allSubArrays(L, L2=None):
        if L2 == None:
            L2 = L[:-1]
        if L == []:
            if L2 == []:
                return []
            return allSubArrays(L2, L2[:-1])
        return [L] + allSubArrays(L[1:], L2)

    def capital_letter(abb, phrase):

        cl = [c for c in abb if c.isupper()]
        try:
            idx = [ i for i,x in enumerate(phrase) if x[0].lower() == cl[0].lower()]
        except:
            return []

        # if len(idx) == 0:
        #     return [None,len(phrase)+1]

        cand = []

        for id in idx:

            mis_matches = 0
            for w in phrase[id:]:
                chr_in_w = False
                for cha in abb:
                    if cha in w:
                        chr_in_w = True
                        break
                if chr_in_w == False:
                    mis_matches += 1
            cand.append((phrase[id:],mis_matches,))

        return cand

    #datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

    lf = []
    for dataset in datasets:
        data = load(dataset)
        lf.extend(abbr_short_long_location_sentences(data))


    # with open('data.txt', 'w') as outfile:
    #     json.dump(lf1,outfile)
    #
    # with open('data.txt', 'r') as json_data:
    #     lf =  json.load(json_data)

    LF_results = []
    for sent in lf:
        tok_set = sent[2].split()

        max_lenght = max(len(sent[0]), sum(map(str.isupper, sent[0]))+3)
        try:
            index = [i for i, x in enumerate(tok_set) if sent[0] in x][0]
        except:
            continue

        if index - max_lenght>0:
            s_start = index - max_lenght
        else:
            s_start = 0

        s_arr = tok_set[s_start:index]

        all_su = allSubArrays(s_arr)
        all_sub = [x for x in all_su if x[-1]==s_arr[-1]]

        # first_char = [''.join([y[0] for y in x]) for x in all_sub]
        # similarity =[]
        # for el in first_char:
        #     similarity.append((difflib.SequenceMatcher(None, sent[0], el).ratio(),Levenshtein.ratio(sent[0], el),nltk.edit_distance(sent[0], el),el))

        res = []
        for candidate in all_sub:
            res.extend(capital_letter(sent[0],candidate))

        all_short = [x for x in res if x[1] == min([x[1] for x in res])]

        if len(all_short) >0:
            ph=' '.join(all_short[0][0])
            LF_results.append((ph,sent[1],sent[0],ph==sent[1]))

    LF_results =list(set(LF_results))
    bads = [x for x in LF_results if not x[3]]
    results = [x for x in LF_results if x[0].split()[-1] == x[1].split()[-1] ]

    acc2 = sum([1 for x in print if x[3]]) / len(print)

    print("Accuracy of long form detection for rules based method:")
    print(acc2)

    test=0

#long form detection - naive NLP
def long_form_detection_tagging(datasets):

    def allSubArrays(L, L2=None):
        if L2 == None:
            L2 = L[:-1]
        if L == []:
            if L2 == []:
                return []
            return allSubArrays(L2, L2[:-1])
        return [L] + allSubArrays(L[1:], L2)

    def capital_letter(abb, phrase):

        cl = [c for c in abb if c.isupper()]
        try:
            idx = [ i for i,x in enumerate(phrase) if x[0].lower() == cl[0].lower()]
        except:
            return []

        # if len(idx) == 0:
        #     return [None,len(phrase)+1]

        cand = []

        for id in idx:

            mis_matches = 0
            for w in phrase[id:]:
                chr_in_w = False
                for cha in abb:
                    if cha in w:
                        chr_in_w = True
                        break
                if chr_in_w == False:
                    mis_matches += 1
            cand.append((phrase[id:],mis_matches,))

        return cand

    lf1 = []
    for dataset in datasets:
        data = load(dataset)
        lf1.extend(abbr_short_long_location_sentences(data))

    with open('data2.txt', 'wb') as outfile:
        _pickle.dump(lf1,outfile)
    with open('data2.txt', 'rb') as json_data:
        lf =  _pickle.load(json_data)

    lf_struct = []
    LF_results = []
    chunks = []
    nounp_list = []
    pos_tagged = []
    for sent in lf:
        tok_set = sent[2].split()



        max_lenght = max(len(sent[0]), sum(map(str.isupper, sent[0]))+3)
        try:
            index = [i for i, x in enumerate(tok_set) if sent[0] in x][0]
        except:
            continue

        grammar = "NP: {<DT>?<JJ>*<NN>}"
        cp = nltk.RegexpParser(grammar)
        res = cp.parse(nltk.pos_tag(tok_set))
        chunks.append((res,sent[0],sent[1]))
        nounp_list.append(([' '.join(leaf[0] for leaf in tree.leaves())
                      for tree in cp.parse(nltk.pos_tag(tok_set[:index])).subtrees()
                      if tree.label()=='NP'],sent[0],sent[1]))

        # try:
        #     tagged = nltk.pos_tag(tok_set[index-max_lenght:index])
        # except:
        #     tagged = nltk.pos_tag(tok_set[0:index])
        # pos_tagged.append(tagged)
        # test = 0


        if index - max_lenght>0:
            s_start = index - max_lenght
        else:
            s_start = 0

        s_arr = tok_set[s_start:index]
        tagged = nltk.pos_tag(s_arr)


        try:
            ind = max([i for i, x in enumerate(tagged) if 'DT' in x[1] or 'IN' in x[1]])

        except:
            ind =0

        long_form = [x for x in s_arr[ind + 1:] if x[0].lower() in sent[0].lower()]

        ph = ' '.join(long_form)
        LF_results.append((ph, sent[1], sent[0], ph == sent[1]))

        logform_tok =sent[1].split() #= nltk.word_tokenize(sent[1])
        t_lf = nltk.pos_tag(logform_tok)
        lf_struct.append(t_lf)

        test = 0

    LF_results =list(set(LF_results))

    RATE = len([x[0][1] for x in lf_struct if  'NN' in x[0][1] or 'JJ' in x[0][1]])/len(lf_struct)
    #print(RATE)
    stopWords = set(nltk.corpus.stopwords.words('english'))

    bads = [x for x in LF_results if not x[3]]
    LF_results = [x for x in LF_results if len(x[0]) > 0]
    res = [x for x in LF_results if x[0].split()[-1] == x[1].split()[-1]]
    acc = sum([1 for x in res if x[3]]) / len(res)

    print("Accuracy of long form detection using naive NLP:")
    print(acc)

    test=0


#generate candidates for Long Fomr detection
def LF_DL(datasets):

    #Supporting function - Generate all subarreys
    def allSubArrays(L, L2=None):
        if L2 == None:
            L2 = L[:-1]
        if L == []:
            if L2 == []:
                return []
            return allSubArrays(L2, L2[:-1])
        return [L] + allSubArrays(L[1:], L2)

    #For each dataset from argument
    for ds in range(1):

        ds1 = datasets
        acronym_meaning = []
        text = []
        text_sent = []

        #list of stop words
        stopWords = set(nltk.corpus.stopwords.words('english'))

        #load dataset from the file and extract acronyms
        for ds in ds1:
            data = load(ds)

            text.extend(get_text(data))
            acronym_meaning.extend(abbr_short_long(data))
        #tokenize by sentence
        for abs in text:
            sent_text = nltk.sent_tokenize(abs)
            text_sent.extend(sent_text)

        #celan up SF-LF pairs
        acronym_meaning = clean_short_long(acronym_meaning)

        training = []
        text_sent= list(set(text_sent))

        #find sentences with pairs SF-LF
        for sent in text_sent:
            for acr in acronym_meaning:
                if acr[0] in sent and acr[1] in sent:
                    training.append([acr,sent])
                # elif acr[0] in sent and acr[1] not in sent:
                #     training.append([acr,sent])

        tr_len = len(training)

        #add 50% of sentences with pairs SF but without its LF
        while tr_len > 0:
            for sent in text_sent:
                if tr_len < 0:
                    break
                for acr in acronym_meaning:
                    if tr_len < 0:
                        break
                    if acr[0] in sent and acr[1] not in sent:
                        training.append([acr, sent])
                        tr_len -= 1
                        print(tr_len)
                        if tr_len < 0:
                            break


        #generate candidates for each sentence
        for sent in training:

            tok_set = sent[1].split()

            #max length of LF
            max_lenght = max(len(sent[0]), sum(map(str.isupper, sent[0][0]))+3)
            try:
                index = [i for i, x in enumerate(tok_set) if sent[0][0] in x][0]
            except:
                continue

            if index - max_lenght>0:
                s_start = index - max_lenght
            else:
                s_start = 0

            # if not sent[0][1] in ' '.join(tok_set[s_start:index]):
            #     continue

            s_arr = tok_set[s_start:index]

            all_su = allSubArrays(s_arr)
            all_sub = [x for x in all_su if x[-1]==s_arr[-1] and x[0] not in stopWords]
            sent.append(all_sub)
        #remove invalid sentences; starting from an acronym
        training2 = [x for x in training if len(x[2]) > 0]

        #generate a feature vector for each candidate
        for sent in training2:

            f_cand = []
            for candidate in sent[2]:
                features = []
                #The ratio of words starting from acronym letters to lenght of acronym
                features.append(sum([1 for x in candidate if x[0].lower() in sent[0][0].lower()])/len(sent[0][0]))

                #The ratio of letters from acronym appearing in order in the full long form:
                aligments = pairwise2.align.globalmx(''.join(candidate).lower(), sent[0][0].lower(), 1, 0)
                features.append(max([x[2] for x in aligments])/len(sent[0][0]))

                #The ratio of letters from acronym appearing in order in the first words of long form:
                aligments2 = pairwise2.align.globalmx(''.join([x[0] for x in candidate]).lower(), sent[0][0].lower(), 1, 0)
                features.append(max([x[2] for x in aligments2]) / len(sent[0][0]))

                #The ratio of # of the letter in acronym to # of words in the long form
                features.append(len(sent[0][0])/len(sent[0][1]))

                #Number of stop words
                features.append(sum([1 for x in candidate if x in stopWords]))

                #The ratio of # of digits in long form to # of digits in the acronym
                try:
                    features.append(sum([1 for x in ''.join(sent[0][1]) if x.isdigit()])/sum([1 for x in sent[0][0] if x.isdigit()]))
                except:
                    features.append(0)

                #number of words not starting with any character in the acronym to the total number of words (not following order in acronym)
                features.append(sum([1 for x in ''.join([x[0] for x in candidate]) if x not in sent[0][0]]))

                # number of words not starting with any character in the acronym to the total number of words (following order in acronym)
                aligments2 = pairwise2.align.globalmx(''.join([x[0] for x in candidate]).lower(), sent[0][0].lower(), 1, 0)
                features.append(len(candidate)-max([x[2] for x in aligments2]))

                # number of words starting with any character in the acronym to the total number of words (not following order in acronym)
                features.append(len(candidate) - features[-2])

                # number of words starting with any character in the acronym to the total number of words (following order in acronym)
                features.append(len(candidate) - features[-2])

                #ratio of stop-words to the total number of words not starting with any character in the acronym
                try:
                    features.append(sum([1 for x in candidate if x.lower() in stopWords])/sum([1 for x in ''.join([x[0] for x in candidate]) if x.lower() not in sent[0][0].lower()]))
                except:
                    features.append(0)

                # ratio of total number of words to the total number of letters in acronym
                features.append(len(candidate)/len(sent[0][0]))

                #number of words
                features.append(len(candidate))

                my_dict = {i: candidate.count(i) for i in candidate}

                #number of duplicate words in substring
                features.append(sum([x for x in my_dict.values() if x> 1]))

                # first word of candidate starts with the first character of acronym
                features.append(1 if candidate[0][0].lower() ==sent[0][0][0].lower() else 0)

                # last word of candidate starts with the last character of acronym
                features.append(1 if candidate[-1][0].lower() == sent[0][0][-1].lower() else 0)

                # does the long form and acronym any punctuation ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
                features.append(1 if any(x in ''.join(candidate) for x in string.punctuation) and any(x in sent[0][0] for x in string.punctuation) else 0)

                # Ratio: number of characters in SF to number of characters in LF
                features.append(len(sent[0][0]) / len(''.join(candidate)))

                # Ratio: number of upper_cases in SF to number of words in LF
                features.append(sum([1 for x in sent[[0][0]] if x.upper()])/len(candidate))

                # levenshtein distance b/w LF and SF
                features.append(jellyfish.levenshtein_distance(''.join(candidate), sent[0][0]))

                # jaro distance b/w LF and SF
                features.append(jellyfish.jaro_distance(''.join(candidate), sent[0][0]))

                # hamming distance b/w LF and SF
                features.append(jellyfish.hamming_distance(''.join(candidate), sent[0][0]))

                # levenshtein distance b/w first letters of LF and SF
                features.append(jellyfish.levenshtein_distance(''.join([x[0] for x in candidate]), sent[0][0]))

                # jaro distance b/w first letters of LF and SF
                features.append(jellyfish.jaro_distance(''.join([x[0] for x in candidate]), sent[0][0]))

                # hamming distance b/w first letters of LF and SF
                features.append(jellyfish.hamming_distance(''.join([x[0] for x in candidate]), sent[0][0]))

                if sent[0][1] == ' '.join(candidate):
                    features.append(1)
                else:
                    features.append(0)


                sent.append(features)




        #save results
        with open(''.join(ds1).replace('.xml',''), 'wb') as outfile:
            _pickle.dump(training2, outfile,protocol=2)
            outfile.close()


datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml","SH_bioc_gold.xml"]
decition_tree_accronym(datasets)
LF_DL(datasets)