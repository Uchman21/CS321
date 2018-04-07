import bioc
import os
import nltk
import lxml
import csv
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from abbreviations import schwartz_hearst
from sklearn import tree
from sklearn import metrics
import sklearn
from sklearn.model_selection import cross_val_score
import matplotlib.colors as colors
import difflib
from fuzzywuzzy import fuzz
import Levenshtein
import json
import _pickle


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('conll2000')


def load(dataset):
    src = os.path.join('data', dataset)
    with open(src) as fp:
        collection = bioc.load(fp)
    return collection

def unique_by_key(elements, key=None):
    if key is None:
        # no key: the whole element must be unique
        key = lambda e: e
    return list({key(el): el for el in elements}.values())

def get_text(collection):
    return [x.passages[1].text for x in collection.documents if x.passages[1].text  != 'No abstract']

def abbr_short_long(collection): #from abstract only w/o title
    annotations = [x.passages[1].annotations for x in collection.documents if len(x.passages[1].annotations) >0]

    short= []
    long = []
    for anno in annotations:
        short.extend([x.text for x in anno if 'SF' in x.id])
        long.extend([x.text for x in anno if 'LF' in x.id])

    annotations_0 = [x.passages[0].annotations for x in collection.documents if len(x.passages[0].annotations) > 0]
    for anno in annotations_0:
        short.extend([x.text for x in anno if 'SF' in x.id])
        long.extend([x.text for x in anno if 'LF' in x.id])

    return [(short[i],long[i]) for i,x in enumerate(short)]

def abbr_short(collection): #from abstract only w/o title
    try:
        annotations = [x.passages[1].annotations for x in collection.documents if len(x.passages[1].annotations) >0]
    except:
        try:
            annotations = [x.passages[0].annotations for x in collection.documents if len(x.passages[0].annotations) > 0]
        except:
            what = 0

    short= []
    for anno in annotations:
        short.extend([x.text for x in anno if 'SF' in x.id])

    annotations_0 = [x.passages[0].annotations for x in collection.documents if len(x.passages[0].annotations) > 0]
    for anno in annotations_0:
        short.extend([x.text for x in anno if 'SF' in x.id])

    return short

def abbr_ids(words,abbr):
    ids = []
    for i, word in enumerate(words):
        if is_abbr(word,abbr):
            ids.append(i)
    return ids

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

def is_abbr_rules(word):
    b = False
    if sum(1 for c in word if c.isupper()) > 2:
        b =True
    return b

def vectors_extend(vectors,i): #i-char before, to extend for j-char after UP TO 2 TODO

    vector_ex = []

    for k,vector in enumerate(vectors):
        if k == 0:
            vector_ex.append([0]*(i) + vector)# + vectors[k+1][:j-1])
        else: #if k < len(vectors) -1:
            vector_ex.append([x for x in vectors[k-1] if x != 0][-i+1:] + [weights[5]] + vector)# + [weights[5]] + vectors[k+1][:j-1])
       # else:
            #vector_ex.append(vectors[k - 1][-i + 1:] + [weights[5]] + vector)# + [weights[5]] + [0]*(j-1))

    return vector_ex

def get_vector(word, weights):
    vec = [0]*n
    for i,c in enumerate(word):
        try:
            vec[i]
        except:
            break

        if c.isalpha():
            if c.isupper():
                vec[i] = weights[0]
            else:
                    vec[i] = weights[1]

        elif c.isdigit():
            vec[i] = weights[2]
        elif c in brackets:
            vec[i] = weights[3]
        else:
            vec[i] = weights[4]

    return vec

def get_vectors(tokens_words, weights):

    vecs = []
    for i,word in enumerate(tokens_words):
        vecs.append(get_vector(word,weights))
    return vecs

def get_tokenize_abstracts(collection):
    raw_text = get_text(data)

    tokens_words = []
    for abstract in raw_text:
        tokens_words.extend(nltk.word_tokenize(abstract))

    test = 0

    return tokens_words

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

def save_list(filename,list):

    thefile = open(filename, 'w')
    for item in list:
        thefile.write("%s\n" % item)

def save_lol_csv(filename,list):

    with open(filename, 'w',newline='') as thefile:
        header = [["id", "word", "extended word", "transformed", "label"]]
        writer = csv.writer(thefile, delimiter='\t')
        header.extend(list)
        writer.writerows(header)

def generate_uniq_tables():
    trans = []
    trans_2 = []
    abbr = []
    datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

    for dataset in datasets:
        data = load(dataset)
        # long = abbr_short_long(data)
        # transformation_list(data)
        abbr.extend(abbr_short(data))  # list of acronyms
        trans.extend(transformation_list(data))
        trans_2.extend(transformation_list_bothside(data))
        # tokens_words = get_tokenize_abstracts(data) #tokenize abstracts
        # ids = abbr_ids(tokens_words,abbr) #indexes of acronyms

        # vec = get_vectors(tokens_words,weights)  #make vectors
        # vecs_ex = vectors_extend(vec,2) #extend vectors

    # save_list("acronym_ids.txt",ids)
    # save_list("vectors.txt",vecs_ex)

    save_lol_csv("transform_all_2char_L.csv", trans)
    save_lol_csv("transform_all_2char_LR.csv", trans_2)

    for i in range(1, 4):
        trans_uniq = unique_by_key(trans, key=itemgetter(i))
        trans_uniq_2 = unique_by_key(trans_2, key=itemgetter(i))

        save_lol_csv("transform_all_2char_L_uniqe" + str(i) + ".csv", trans_uniq)
        save_lol_csv("transform_all_2char_LR_uniqe" + str(i) + ".csv", trans_uniq_2)

        print("i=" + str(i))
        print("lenght, L = " + str(len(trans_uniq)) + " LR= " + str(len(trans_uniq_2)))
        print("L, T= " + str(len([x for x in trans_uniq if x[4]])) + " F = " + str(
            len([x for x in trans_uniq if not x[4]])))
        print("LR, T= " + str(len([x for x in trans_uniq_2 if x[4]])) + " F = " + str(
            len([x for x in trans_uniq_2 if not x[4]])))

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

def abb_cleaning(abb, max_length):

    abb = list(set(abb))
    abb = [x for x in abb if x[0][1].length <= max_length]
    abb = [x for x in abb if  x[0][1].length < x[1][1].length]
    abb = [x for x in abb if len(x[0][0]) < len(x[1][0])]

    abb = [x for x in abb if len(x[0][0]) == x[0][1].length]
    abb = [x for x in abb if len(x[1][0]) == x[1][1].length]

    return abb

def sf_lf_stats_abb():
    trans = []
    trans_2 = []
    abbr = []
    datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

    for dataset in datasets:
        data = load(dataset)
        abbr.extend(abbr_short_long_location(data))

    abbr = abb_cleaning(abbr,1000)
    lengths_SF = [x[0][1].length for x in abbr]
    lengths_LF = [x[1][1].length for x in abbr]
    lengths_LF_words = [len(x[1][0].split()) for x in abbr]

    offset_SF = [x[0][1].offset for x in abbr]
    offset_LF = [x[1][1].offset for x in abbr]

    long_SF = [x for x in abbr if x[0][1].length > 10]
    long_LF = [x for x in abbr if x[1][1].length > 50]

    distance_after = [x[0][1].offset - x[1][1].length - x[1][1].offset for x in abbr if (x[0][1].offset - x[1][1].offset) >0]
    distance_before = [x[1][1].offset - x[1][1].length - x[0][1].offset for x in abbr if (x[1][1].offset - x[0][1].offset) > 0]

    for i in range (0,11):
        rate = len([x for x in abbr if min( np.ceil((x[0][1].length)*2),(x[0][1].length)+5) >= len(x[1][0].split())])/len(abbr)
        new_rate = len([x for x in abbr if len(x[1][0].split()) <= max(x[0][1].length,sum(1 for c in x[0][0] if c.isupper()) + i)])/len(abbr)
        #print("i= " + str(i) + ", rate= " + str(round(rate, 4)) + ", NEW_rate= " + str(round(new_rate, 4)))
        #print("i= " + str(i) + ", rate= " + str(round(new_rate, 4)))

        old_error = np.max([min(np.ceil((x[0][1].length)*2),(x[0][1].length)+5) - len(x[1][0].split()) for x in abbr if min( np.ceil((x[0][1].length)*2),(x[0][1].length)+5) >= len(x[1][0].split())])
        new_error = np.max([max(x[0][1].length,sum(1 for c in x[0][0] if c.isupper())+i) - len(x[1][0].split()) for x in abbr if len(x[1][0].split()) <= max(x[0][1].length,sum(1 for c in x[0][0] if c.isupper()) + i)])
        print("i= " + str(i) + ", OLD_error= " + str(round(old_error, 4)) + ", NEW_error= " + str(round(new_error, 4)))

    #X,Y = np.meshgrid(lengths_SF, lengths_LF_words)
    # rates = []
    # for i in range(0, 11):
    #     rate = len([x for x in abbr if sum(1 for c in x[0][0] if c.isupper()) == i]) / len(abbr)
    #     print("i= " + str(i) +  ", rate= " + str(round(rate,4)))
    #     rates.append(rate)
    #
    # upper = [len([x for x in abbr if sum(1 for c in x[0][0] if c.isupper()) <= i])/len(abbr) for i in range(0,11)]
    #
    # plt.figure()
    # hu = plt.bar(range(len(upper)),upper,1/1.5)
    # plt.xlabel('<= #UpperCases')
    # plt.ylabel('%')
    # plt.title('n and less upper cases.')
    #
    # plt.figure()
    # upp = plt.bar(range(len(rates)),rates,1/1.5)
    # plt.title('Percent of acronyms containing exactly n upper cases.')
    # plt.ylabel('%')
    # plt.xlabel('n upper cases')
    #
    cor= np.corrcoef(lengths_LF_words,lengths_SF)[0, 1]

    old_error = [min(np.ceil((x[0][1].length) * 2), (x[0][1].length) + 5) - len(x[1][0].split()) for x in abbr if
                        min(np.ceil((x[0][1].length) * 2), (x[0][1].length) + 5) >= len(x[1][0].split())]

    new_error = [max(x[0][1].length, sum(1 for c in x[0][0] if c.isupper()) + 3) - len(x[1][0].split()) for x in abbr if
         len(x[1][0].split()) <= max(x[0][1].length, sum(1 for c in x[0][0] if c.isupper()) + 3)]

    plt.figure()
    old = plt.boxplot(old_error)
    plt.title("Old metric")

    plt.figure()
    old = plt.boxplot(new_error)
    plt.title("New metric")

    # plt.figure()
    # box = plt.boxplot(lengths_SF)
    # #plt.title("Distance (in characters) b/w LF and SF")
    # plt.title("# of characters in Short Form (SF)")
    #
    # #l = plt.plot(lengths_SF,lengths_LF_words,'bo')
    # plt.figure()
    # h = plt.boxplot(lengths_LF_words)
    # plt.title("# of words in Long Form (LF)")
    #
    # plt.figure()
    # weights = np.ones_like(lengths_SF) / float(len(lengths_SF))
    # box = plt.hist(lengths_SF,bins='auto',rwidth=0.7)
    # # plt.title("Distance (in characters) b/w LF and SF")
    # plt.title("# of characters in Short Form (SF)")
    #
    # # l = plt.plot(lengths_SF,lengths_LF_words,'bo')
    # plt.figure()
    # weights = np.ones_like(lengths_LF_words) / float(len(lengths_LF_words))
    # h = plt.hist(lengths_LF_words,bins='auto',rwidth=0.7)
    # plt.title("# of words in Long Form (LF)")
    #
    # plt.figure()
    # twoD = plt.hist2d(lengths_SF, lengths_LF_words,bins=range(1,15),norm=colors.LogNorm())
    # plt.colorbar()
    # plt.xlabel("# of characters in SF")
    # plt.ylabel("# of words in LF")
    # plt.title("2D histogram, # of characters in SF / # of words in LF  (LogNorm)")
    #
    #
    # plt.figure()
    # twoD = plt.hist2d(lengths_SF, lengths_LF_words, bins=range(1, 15))
    # plt.xlabel("# of characters in SF")
    # plt.ylabel("# of words in LF")
    # plt.title("2D histogram, # of characters in SF / # of words in LF ")
    # plt.colorbar()
    #
    # plt.figure()
    # twoD = plt.hist2d(lengths_SF, lengths_LF, norm=colors.LogNorm())
    # plt.colorbar()
    # plt.xlabel("# of characters in SF")
    # plt.ylabel("# of characters in LF")
    # plt.title("2D histogram, # of characters in SF / # of words in LF  (LogNorm)")
    #
    # plt.figure()
    # twoD = plt.hist2d(lengths_SF, lengths_LF)
    # plt.xlabel("# of characters in SF")
    # plt.ylabel("# of characters in LF")
    # plt.title("2D histogram, # of characters in SF / # of words in LF ")
    # plt.colorbar()
    plt.show()



    test = 0

def decition_tree_accronym():

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

    datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

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

def long_form_detection():

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

    datasets = ["medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

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

def long_form_detection_tagging():

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

    datasets = ["medstract_bioc_gold.xml"]#, "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"]

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

#Datasets included:
#"medstract_bioc_gold.xml", "Ab3P_bioc_gold.xml", "bioadi_bioc_gold.xml", "SH_bioc_gold.xml"
#download from https://sourceforge.net/p/bioc/blog/2014/06/i-am-looking-for-the-abbreviation-definition-corpus-where-can-i-find-it/
#and place them in /data folder

#Each method output classification  score on all datasets

#Acronym detection using Decision Trees, results of 5-fold CV:
#decition_tree_accronym()

#Long form detection

#Method #1 - simple rules:
#long_form_detection()

#Method #2 - naive NLP
#long_form_detection_tagging()
