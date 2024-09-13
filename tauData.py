# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 05:18:36 2022
taudata Library for snScrape twitter scraper
@author: Taufik Sutanto
"""
import warnings; warnings.simplefilter('ignore')
import pymysql, re, itertools, pandas as pd, pickle
import numpy as np, time, sys, os, zipfile#, datetime as dm
from sqlalchemy import create_engine
from textblob import TextBlob
from textblob import Word
from html import unescape
from unidecode import unidecode
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import mysql.connector as sql
import folium #, plotly.express as px
import plotly.graph_objects as go

lemma_id = StemmerFactory().create_stemmer()
slangS = pd.read_csv('/content/BERT/slang.csv', header=None, index_col=0).squeeze().to_dict()
factory = StopWordRemoverFactory()
stopWordsID = set(factory.get_stop_words())
stopWordsEN = set(stopwords.words("english"))

def compress(file_, ext="csv", level=9, delete=True):
    zipfile.ZipFile(file_.replace(".{}".format(ext), ".zip"), 'w', compression=zipfile.ZIP_BZIP2, compresslevel=level).write(file_)
    if delete:
        os.remove(file_)
    return True

def loadText(f_):
    ff_=open(f_,"r",encoding="utf-8", errors='replace')
    sw = ff_.readlines(); ff_.close()
    return sw

stopCustom = set([s.lower().strip() for s in loadText('/content/BERT/stopwords_id.txt')])
stopWordsID = stopWordsID.union(stopCustom)
stopCustom = 'desember des jan january januari sep september feb february \
    februari subscribe named video view all wa link'.split()
for w in stopCustom:
    stopWordsID.add(w)

def loadCorpus(lan = 'en'):
    corpus = set()
    if lan.lower().strip() in ['en', 'english', 'inggris', 'eng']:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import brown
        try:
            f = open('/content/BERT/corpusEN.pckl', 'rb')
            corpus = pickle.load(f); f.close()
        except:
            categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
             'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
             'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
             'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
            data = fetch_20newsgroups(categories=categories,remove=('headers', 'footers', 'quotes'))
            news = [doc for doc in data.data]
            for i, r in enumerate(news):
                news[i] = cleanText(r, min_charLen = 3, max_charLen = 18, stopFilter=False, fixSlang=False)
            binary_vectorizer = CountVectorizer(binary = True)
            _ = binary_vectorizer.fit_transform(news)
            corpus = set([v for v,k in binary_vectorizer.vocabulary_.items()])
            Brown = set([str(t).lower().strip() for t in brown.words()])
            corpus.update(Brown)
            corpus.update(stopWordsEN)
            f = open('/content/BERT/corpusEN.pckl', 'wb')
            pickle.dump(corpus, f); f.close()
    elif lan.lower().strip() in ['id', 'indonesia', 'indonesian']:
        try:
            f = open('/content/BERT/corpusID.pckl', 'rb')
            corpus = pickle.load(f); f.close()
        except:
            corpus = stopWordsID
            sw = loadText('/content/BERT/stopwords_id.txt')
            sw = set([w.lower().strip() for w in sw])
            corpus = corpus.union(sw)
            corpus = corpus.union(slangS.values())
            corpus = corpus.union(slangS.keys())
            kd = loadText('/content/BERT/kata_dasar_id.txt')
            kd = set([k.split()[0].lower().strip() for k in kd])
            corpus = corpus.union(kd)
            pos = set([k.lower().strip() for k in loadText('/content/BERT/kataPosID.txt')])
            corpus = corpus.union(pos)
            neg = set([k.lower().strip() for k in loadText('/content/BERT/kataNegID.txt')])
            corpus = corpus.union(neg)
            mtc = []
            for k in loadText('/content/BERT/Indonesian_Manually_Tagged_Corpus.tsv'):
                try:
                    w = k.split('\t')[0].lower().strip()
                    if len(w)>2:
                        mtc.append(w)
                except:
                    pass
            mtc = set([tok for tok in mtc if sum([1 for d in tok if d.isalpha()])==len(tok)])
            corpus.union(mtc)
            f = open('/content/BERT/corpusID.pckl', 'wb')
            pickle.dump(corpus, f); f.close()
    else:
        print('Language not supported')
        return None
    return corpus

corpusID = loadCorpus(lan = 'id')
politicsCorpus = 'budiman prabowo subianto anies baswedan muhaimin iskandar \
    gibran rakabuming mahfud mahfudmd ganjar pranowo pilpres jokowi\
        orangsolo ngamen kaesang gerindra golkar pdip pks nasdem alhamdulillah\
            ridwan kamil ganjarpranowo agung ummat makassar dpc pasuruan bogor\
               kpk kpu najwa shihab bapack insya timnas polri kapolri netralitas babak jawa timur\
               divhumas polri pileg pkssejahtera gaul sulsel kenang degan cermat tohir alkes banten pamulang\
               sidoarjo ikn ruu gaspol cawapres gus survei survei elektabilitas jusuf kalla tkn\
               ponpes sholawat tribun banyuwangi pendekar grebeg'.split()
for w in politicsCorpus:
    corpusID.add(w)    
corpusEN = loadCorpus(lan = 'en')

def noVocal(t):
    V = {'a', 'b', 'c', 'd', 'e'}
    for v in V:
        if v in t:
            return False
    return True  

isascii = lambda s: len(s) == len(s.encode())
def nameXpander(t, hyphen, preProcess=False, keepSource=False):
    """
    - https://pypi.org/project/PyHyphen/
    - from hyphen.dictools import *
    - install("id_ID")
    #from hyphen import Hyphenator; hyphen = Hyphenator('id_ID')
    """
    T = t
    if preProcess:
        T = unidecode(T)
        T = ' '.join(re.split("(\d+)", T)) # Pisah kalau ada angka
        T = T.title()
        pisahBesarKecil = re.compile(r'[A-Z][^A-Z]*')
        T = ' '.join(re.findall(pisahBesarKecil, T))
        T = T.lower().strip()
        T = ''.join(''.join(s)[:2] for _, s in itertools.groupby(T)) # remove repetition
        T = re.sub(r'[^.,_a-zA-Z0-9 -\.]','', T)
        T = " ".join([t for t in TextBlob(T).words if len(t)>1])
    
    th = hyphen.syllables(t)
    th = [syl for syl in th if len(syl)>1 and isascii(syl)]
    if keepSource:
        return T + ' ' + ' '.join(th).lower()
    else:
        return ' '.join(th).lower()

def cleanText(T, min_charLen=2, max_charLen=18, stopFilter=True, maxWords=0, fixSlang=True, excludeMix=True, lang='id', lemma=False, extraStop=False):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    listKata = re.sub(r'[^.,_a-zA-Z0-9 -\.]','',t)
    listKata = TextBlob(listKata).words
    if excludeMix:
        listKata = [tok for tok in listKata if sum([1 for d in tok if d.isdigit()])==0] # mix of number and char not included
        
    if fixSlang:
        for i, kata in enumerate(listKata):
            if kata in slangS.keys():
                listKata[i] = slangS[kata] 
           
    if lemma and lang=='id':
        listKata = lemma_id.stem(' '.join(listKata).replace("'","").replace('"','')).split()
    elif lemma and lang=='en':
        listKata = [Word(w.replace("'","").replace('"','')).lemmatize() for w in listKata]
        
    if stopFilter and lang=='id':
        listKata = [tok for tok in listKata if (str(tok) not in stopWordsID)]
        if extraStop:
            listKata = [tok for tok in listKata if (str(tok) not in stopWordsEN)]
    elif stopFilter and lang=='en':
        listKata = [tok for tok in listKata if (str(tok) not in stopWordsEN)]
    
    listKata = [tok for tok in listKata if len(str(tok))>=min_charLen and len(str(tok))<=max_charLen] 
    if maxWords>0:
        return ' '.join(listKata[:maxWords])
    else:
        return ' '.join(listKata)# Return kalimat lagi

def checkLanguage(txt, thr=0.5):
    T = txt.split()
    id_ = sum([1 for t in T if t in corpusID])/len(T) # banyaknya kata bahasa Indonesia
    en_ = sum([1 for t in T if t in corpusEN])/len(T) # banyaknya kata bahasa Indonesia
    if id_>thr and id_>en_:
        return 'id', id_ # , s_/len(T) is the probability/support
    elif en_>thr and en_>id_:
        return 'en', en_ # , s_/len(T) is the probability/support
    else:
        return 'unknown', 1-(id_+en_)/2 # , s_/len(T) is the probability/support

def tauSleep(nSleep, verbose_=True, rand_=False, inc_=1):
    if rand_:
        tSleep = max(inc_, int(nSleep*np.random.rand()))
    else:
        tSleep = nSleep
    if verbose_:
        print("Sleeping", end=' ', flush=True)
        for i in range(int(tSleep/inc_)):
            time.sleep(1)
            print("*", end=' ', flush=True)
        print(flush=True)
    else:
        time.sleep(tSleep)

def fResume(var=None, file='resume_var.pckl', action='save'):
    if action.lower().strip()=='save':
        with open(file, 'wb') as f:  
            pickle.dump(var, f)
        return True
    elif action.lower().strip()=='load':
        with open(file, 'rb') as f:
            var = pickle.load(f)
            return var
    else:
        sys.exit("Error! action parameter should be 'save' or 'load'")
        return None

def conMysql(dbPar, maxTry=7, verbose=False):
    try_ = 0 # dbPar = {'db_': 'rpi', 'usr':'root', 'pas':'', 'hst':'localhost'}
    while try_<maxTry:
        try:
            con =  pymysql.connect(host=dbPar['host'],user=dbPar['user'],passwd=dbPar['pass'],db=dbPar['db_'])
            if verbose:
                with con.cursor() as cur:
                    cur.execute('SELECT VERSION()')
                    version = cur.fetchone()
                    print(f'Connected! Current Database version: {version[0]}')
            return con
        except (pymysql.Error) as e:      
            print ("Error Connecting to MySQL %d: %s \n Retrying after 3 seconds ... " % (e.args[0],e.args[1]))
            try_ += 1; time.sleep(3)
            
def execQryOld(qry, dbPar, values=False, maxTry=7):
    try_, nSleep = 0, 1
    while try_<maxTry:
        try:
            con = conMysql(dbPar); cur = con.cursor()
            if values:
                cur.executemany(qry, values)
            else:
                cur.execute(qry)
            if "optimize table " not in qry.lower().strip():
                con.commit()
            try:
                con.close()
                del con, cur
            except:
                pass
            return True
        except Exception as err_:
            print('query Error =\n{}'.format(err_))
            try_ += 1
            tauSleep(nSleep*try_, verbose_=True, rand_=False, inc_=1)
    return False

def getData(dbPar, tbl, cols='*', val_=None, col_=None, limit_=None, maxTry=7):
    try_ = 0
    if val_:
        qry  = "SELECT {} FROM {} WHERE {}='{}'".format(cols, tbl, col_, val_)
    else:
        qry  = "SELECT {} FROM {}".format(cols, tbl)
    if limit_:
        qry = "{} LIMIT {}".format(qry, limit_)
    while try_<maxTry:
        try:
            db = conMysql(dbPar); cur = db.cursor()
            cur.execute(qry)
            data = cur.fetchall()
            try:
                cur.close();db.close()
                del cur, db
            except:
                pass
            if data:
                return data
            else:
                return False
        except Exception as err_:
            print('query Error =\n{}'.format(err_))
            try_ += 1
            time.sleep(3)
    return None

def df2MySQL(df, dbPar, tbl, maxTry=7, batch=30, verbose=False, update=False):
    t = "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=dbPar['host'], db=dbPar['db_'], user=dbPar['user'], pw=dbPar['pass'])
    conn = create_engine(t)
    if update:
        t = df.to_sql(name=tbl, con=conn, if_exists='replace', index=False)
    else:
        t = df.to_sql(name=tbl, con=conn, if_exists='append', index=False)
    conn.dispose(); del conn
    return t

def execQry(dbPar, qry=None, verbose=False):
    conn = sql.connect(db=dbPar["db_"], user=dbPar["user"], 
                   host=dbPar["host"], password=dbPar["pass"], use_unicode=True, 
                   charset= "utf8", auth_plugin= "mysql_native_password", autocommit=True)
    if qry:
        mycursor = conn.cursor()
        mycursor.execute(qry)
        #if "optimize table " not in qry.lower().strip():
        #    conn.commit()
        try:
            conn.commit()
        except:
            pass
        if mycursor.rowcount>0 and verbose:
            print(mycursor.rowcount, "record(s) affected")
            try:
                conn.commit()
            except:
                pass
            conn.close(); del conn
            return mycursor.rowcount
    else:
        return conn
    
def plot_line_go_graph(df,col_x,col_y,col_color = None,col_filter = None,add_points = False, title='', dd="All Social Media") :
    df_graph = df.copy()
    if add_points :
        param_mode='lines+markers'
        param_name='lines+markers'
    else :
        param_mode='lines'
        param_name='lines'
    fig = go.Figure()
    if col_filter is None :
        if col_color is None :
            fig.add_trace(go.Scatter(x=df_graph[col_x], y=df_graph[col_y],mode=param_mode,name=param_name))
        else :
            for c in df_graph[col_color].unique() :
                fig.add_trace(go.Scatter(x=df_graph[df_graph[col_color]==c][col_x], y=df_graph[df_graph[col_color]==c][col_y],mode=param_mode,name=c))
    else :
        df_graph[col_filter] = df_graph[col_filter].fillna("NaN")
        if col_color is None :
            L_filter = []
            for f in df_graph[col_filter].unique():
                fig.add_trace(go.Scatter(x=df_graph[df_graph[col_filter]==f][col_x], y=df_graph[df_graph[col_filter]==f][col_y],mode=param_mode,name=param_name,visible = False))           
                L_filter.append(f)
            df_graph_gb = df_graph.groupby([col_x],as_index=False).agg({col_y:"sum"})
            fig.add_trace(go.Scatter(x=df_graph_gb[col_x], y=df_graph_gb[col_y],mode=param_mode,name=param_name,visible = True))
            L_filter.append("ALL Data")
        else :
            L_filter = []
            for group, df_group in df_graph.groupby([col_color, col_filter]):
                fig.add_trace(go.Scatter(
                    x=df_group[col_x], 
                    y=df_group[col_y],
                    mode=param_mode,
                    name=group[0],
                    visible=False
                ))
                L_filter.append(group[1])
                
            df_graph_gb = df_graph.groupby([col_x,col_color],as_index=False).agg({col_y:"sum"})
            for clr in df_graph_gb[col_color].unique() :
                fig.add_trace(go.Scatter(
                        x=df_graph_gb[df_graph_gb[col_color]==clr][col_x], 
                        y=df_graph_gb[df_graph_gb[col_color]==clr][col_y],
                        mode=param_mode,
                        name=clr,
                        visible=True
                    ))
                L_filter.append(dd)
                
        updatemenu = []
        buttons = []
        for b in [dd] + list(df_graph[col_filter].unique()) :
            visible_traces = [True if b == i else False for i in L_filter]
            buttons.append(dict(method='restyle',
                                label=b.title(),
                                visible=True,
                                args=[{'visible' : visible_traces}]
                            ))
        # some adjustments to the updatemenus
        updatemenu = []
        your_menu = dict()
        updatemenu.append(your_menu)
        updatemenu[0]['buttons'] = buttons
        updatemenu[0]['direction'] = 'down'
        updatemenu[0]['showactive'] = True
        updatemenu[0]['xanchor']='left'
        updatemenu[0]['yanchor']='top'
        updatemenu[0]['x']=.75
        updatemenu[0]['y']=1.1
        # add dropdown menus to the figure
        fig.update_layout(updatemenus=updatemenu)
        if col_color is None :
            fig.update_layout(showlegend=False)
    
    fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        },
        hoverlabel=dict(
            #bgcolor="white", 
            font_size=12, 
            #font_family="Rockwell"
        ),
        hovermode = "x"
    )
    fig.update_xaxes(showspikes=True, spikecolor = 'black', showline=True, linewidth=1,linecolor='black', ticks = "outside", tickwidth = 1, tickcolor = 'black',ticklen = 5)
    fig.update_yaxes(showspikes=True, spikecolor = 'black', showline=True, linewidth=1,linecolor='black', ticks = "outside", tickwidth = 1, tickcolor = 'black',ticklen = 5)
    #fig.update_layout(title_text=title, title_x=0.5)
    return fig
    
def generateBaseMap(default_location=[-0.789275, 113.921], default_zoom_start=5):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

if __name__ == '__main__':
    pass
    """
    def df2MySQL_OLD(df, dbPar, tbl, maxTry=7, batch=30):
        qry = 'INSERT INTO {} ('.format(tbl)
        qry = qry + ', '.join(df.columns) + ') VALUES '
        qry = qry + "("+', '.join(["%s"]*len(df.columns))+")"
        values = []
        for i, d in df.iterrows():
            values.append([str(dt) for dt in d.tolist()])
            if len(values)>=batch:
                execQry(qry, dbPar, values=values, maxTry=maxTry)
                values = []
        if len(values)>0:
            execQry(qry, dbPar, values=values, maxTry=maxTry) 
        return True
    """
