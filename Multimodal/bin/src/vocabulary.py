import pandas as pd
from nltk.tokenize import word_tokenize

class Vocabulary:

    def __init__( self, vocab_path ):
        self.path = vocab_path
        self.df = pd.read_csv( self.path )

    def __len__( self ):
        return self.df.shape[0] + 1

    def vocab_index( self, token ):
        try:
            return int( self.df[ self.df.words==token ].index[0] ) + 1
        except:
            return int( self.df[ self.df.words=='<unk>' ].index[0] ) + 1

    def get_caption_ids( self, caption ):
        ids = []
        ids.append( self.vocab_index( '<s>') )
        clean_caption = []
        for token in word_tokenize( caption.lower() ):
            if token.isalnum():
                idx = self.vocab_index( token )
                ids.append( idx )
                clean_caption.append( token )
        ids.append( self.vocab_index( '</s>' ) )
        clean_caption = ' '.join( clean_caption )
        return clean_caption, ids

    def get_caption_from_index( self, id ):
        try:
            return self.df.words.loc[ id-1 ]
        except:
            return '<unk>'

    def arrays_to_sentences( self, ys ):
        sentences = []
        for arr in ys:
            temp = []
            for id in arr:
                word = self.get_caption_from_index( id )
                if word == '</s>':
                    break
                temp.append( word )
            sentences.append( temp )
        return sentences
