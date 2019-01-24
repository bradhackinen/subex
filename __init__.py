import os
import pandas as pd
from pandasUtilities import dfChunks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torchUtilities import *
from scipy.ndimage.measurements import label, find_objects
from ast import literal_eval
import seaborn as sb
import matplotlib.pyplot as plt



def matchSplitter(s):
    for t in s.split('\n'):
        return t.strip()


def loadTrainingStrings(csvFile,match_sep=['\n','\r'],encoding='utf8'):
    trainingStringsDF = pd.read_csv(csvFile,encoding=encoding,na_filter=False)
    assert ('string' in trainingStringsDF.columns) & ('matches' in trainingStringsDF.columns)
    assert trainingStringsDF['string'].isnull().sum() == 0
    if match_sep:
        for sep in match_sep[1:]:
            trainingStringsDF['matches'] = trainingStringsDF['matches'].str.replace(sep,match_sep[0])

        trainingStringsDF['matches'] = trainingStringsDF['matches'].fillna('').astype(str)
        trainingStringsDF['matches'] = trainingStringsDF['matches'].apply(lambda s: tuple(filter(bool,s.split(match_sep[0]))))
    else:
        trainingStringsDF['matches'] = trainingStringsDF['matches'].apply(literal_eval)

    return trainingStringsDF


def saveTrainingStrings(trainingStringsDF,csvFile,match_sep='\n',encoding='utf8'):
    trainingStringsDF = trainingStringsDF.copy()
    assert set(trainingStringsDF.columns) == {'string','matches'}
    assert trainingStringsDF['string'].isnull().sum() == 0
    if match_sep:
        trainingStringsDF['matches'] = trainingStringsDF['matches'].apply(lambda matches: match_sep.join(matches))

    trainingStringsDF.to_csv(csvFile,encoding=encoding,index=False)


def charLabel(stringChars,matchChars):
    labels = torch.zeros(len(stringChars))
    for match in matchChars:
        start = stringChars.find(match)
        if start < 0:
            print('Warning: Match chars {} not found in string chars {}'.format(match,stringChars))
        end = start + len(match)
        labels[start:end] = 1
    return labels


def buildTrainingData(trainingStringsDF,max_len=500):
    trainingDF = trainingStringsDF.copy()
    trainingDF['chars'] = trainingDF['string'].apply(lambda s:stringToAscii(s)[:max_len])
    trainingDF['match_chars'] = trainingDF['matches'].apply(lambda m:tuple(stringToAscii(s)[:max_len] for s in m))

    trainingDF['labels'] = [charLabel(s,m) for i,s,m in trainingDF[['chars','match_chars']].itertuples()]

    return trainingDF[['chars','labels']]


def packMinibatchData(chars,labels):
    charLabels = {s:y for s,y in zip(chars,labels)}

    with torch.no_grad():
        # Pack chars
        packedChars,chars = bytesToPacked1Hot(chars,clamp_range=(31,126))

        # Align labels
        labels = [charLabels[s] for s in chars]

        # Pack labels
        paddedLabels = nn.utils.rnn.pad_sequence(labels).t().unsqueeze(2)
        packedLabels = nn.utils.rnn.pack_padded_sequence(paddedLabels,[len(y) for y in labels],batch_first=True)

    return packedChars,packedLabels


def generateMinibatches(trainingDataDF,size=10,cuda=False,max_len=200,shuffle=True):
    if shuffle:
        trainingDataDF = trainingDataDF.sample(frac=1)

    for minibatchDF in dfChunks(trainingDataDF,size):

        packedChars,packedLabels = packMinibatchData(minibatchDF['chars'],minibatchDF['labels'])

        if cuda:
            packedChars.data.pin_memory()
            packedLabels.data.pin_memory()

            packedChars = packedToCuda(packedChars)
            packedLabels = packedToCuda(packedLabels)

        yield packedChars,packedLabels


class charClassifier(nn.Module):
    def __init__(self,d_in=96,d_out=1,d_hidden=100,layers=3,bidirectional=True):
        super().__init__()
        self.char_embedding = nn.Linear(d_in,d_hidden)
        self.embedding_dropout = nn.Dropout()
        self.gru = nn.GRU(d_hidden,d_hidden,layers,bidirectional=bidirectional,batch_first=True,dropout=0.5)
        # self.out_dropout = nn.Dropout()
        self.out = nn.Linear(d_hidden*(1+int(bidirectional)),d_out)

    def forward(self,W):
        X = PackedSequence(self.embedding_dropout(self.char_embedding(W.data)),W.batch_sizes)
        H,h = self.gru(X)
        Y = PackedSequence(self.out(H.data),W.batch_sizes)

        return Y


def newModel(cuda=False,d=100,layers=2,bidirectional=True,lr=1e-3,weight_decay=1e-6):
    modelPackage = {'args':locals(),'loss_history':pd.DataFrame()}

    modelPackage['model'] = charClassifier(d_hidden=d,layers=layers,bidirectional=bidirectional)

    if cuda:
        modelPackage['model'] = modelPackage['model'].cuda()

    modelPackage['params'] = {'params':modelPackage['model'].parameters(),'lr':lr,'weight_decay':weight_decay}
    modelPackage['optimizer'] = torch.optim.Adam([modelPackage['params']])

    return modelPackage


def saveModelPackage(modelPackage,filename):
    state = {
    'args':modelPackage['args'],
    'loss_history':modelPackage['loss_history'],
    'model_state': modelPackage['model'].state_dict(),
    'optimizer_state': modelPackage['optimizer'].state_dict(),
    }
    torch.save(state,filename)


def loadModelPackage(filename):
    state = torch.load(filename)

    modelPackage = newModel(**state['args'])
    modelPackage['model'].load_state_dict(state['model_state'])
    modelPackage['optimizer'].load_state_dict(state['optimizer_state'])

    modelPackage['args'] = state['args']
    modelPackage['loss_history'] = state['loss_history']

    return modelPackage


def trainMinibatch(batchData,modelPackage):
    packedChars,packedLabels = batchData

    modelPackage['model'].train()

    packedOutput = modelPackage['model'](packedChars)

    loss = F.binary_cross_entropy_with_logits(packedOutput.data,packedLabels.data,size_average=False)

    modelPackage['optimizer'].zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(modelPackage['model'].parameters(),5)

    modelPackage['optimizer'].step()

    return {'size':float(packedChars.batch_sizes[0].cpu()),'loss':float(loss.data.cpu())}


def trainModel(modelPackage,trainingStringsDF,testStringsDF=None,save_as=None,minibatch_size=10,epochs=10,exit_function=None,cuda=False,verbose=False):
    trainingDataDF = buildTrainingData(trainingStringsDF)

    cuda = next(modelPackage['model'].parameters()).is_cuda

    try:
        b_start,b_end = minibatch_size
        b_schedule = np.linspace(b_start,b_end,epochs).astype(int)
    except:
        b_schedule = minibatch_size*np.ones(epochs).astype(int)

    # Train epochs
    for i in range(epochs):
        bar_freq = max(1,int((len(trainingDataDF)/b_schedule[i])/100))

        epochBatchIterator = (generateMinibatches(trainingDataDF,cuda=cuda,size=b_schedule[i]),)
        modelPackage['loss_history'] = trainWithHistory(lambda b: trainMinibatch(b,modelPackage),epochBatchIterator,modelPackage['loss_history'],
                                        exit_function=exit_function,verbose=verbose,bar_freq=bar_freq)

        if verbose and testStringsDF is not None:
            testMatchesDF = scoreTestStringsDF(testStringsDF,modelPackage)
            print('Test loss: {:.3f}, char accuracy: {:.3f}, fraction correct: {:.3f}'.format(testMatchesDF['loss'].mean(),testMatchesDF['accuracy'].mean(),testMatchesDF['correct'].mean()))

        if save_as is not None:
            if verbose: print('Saving model as {}'.format(save_as))
            saveModelPackage(modelPackage,save_as)

    return modelPackage['loss_history']


def matchesFromProbs(chars,probs,threshold=0.5):
    labels,n = label(probs > threshold)
    matches = [chars[span[0]].decode('ascii') for span in find_objects(labels)]
    return matches


def extract(strings,modelPackage,batch_size=50,return_char_scores=False,dropout_samples=0):
    resultsDF = pd.DataFrame(list(strings),columns=['string'])

    uniqueDF = resultsDF.drop_duplicates()
    uniqueDF['chars'] = uniqueDF['string'].apply(stringToAscii)
    uniqueDF['len'] = [len(c) for c in uniqueDF['chars']]
    uniqueDF = uniqueDF[uniqueDF['len']>0]

    batchResults = []
    for batchDF in dfChunks(uniqueDF,batch_size):
        batchDF = batchDF.sort_values('len',ascending=False)

        with torch.no_grad():
            packedChars,_ = bytesToPacked1Hot(list(batchDF['chars']),clamp_range=(31,126),presorted=True)

            if next(modelPackage['model'].parameters()).is_cuda:
                packedChars = packedToCuda(packedChars)

            # Compute point estimates (no dropout)
            modelPackage['model'].eval()

            packedOutput = modelPackage['model'](packedChars)

            packedProbs = PackedSequence(F.sigmoid(packedOutput.data),packedOutput.batch_sizes)
            paddedProbs,lengths = torch.nn.utils.rnn.pad_packed_sequence(packedProbs)

            packedEntropies = PackedSequence(F.binary_cross_entropy_with_logits(packedOutput.data,packedProbs.data,reduce=False),packedOutput.batch_sizes)
            paddedEntropies,lengths = torch.nn.utils.rnn.pad_packed_sequence(packedEntropies)

            batchDF['probs'] = [x[:l].ravel() for x,l in zip(paddedProbs.t().cpu().numpy(),lengths)]
            # batchDF['entropies'] = [x[:l].cpu().numpy() for x,l in zip(paddedEntropies.t(),lengths)]
            batchDF['entropy'] = [x[:l].sum() for x,l in zip(paddedEntropies.t().cpu().numpy(),lengths)]

            batchDF['matches'] = [tuple(matchesFromProbs(c,p)) for i,c,p in batchDF[['chars','probs']].itertuples()]

            # Estimate uncertainty using dropout samples (if dropout samples > 0)
            if dropout_samples:

                samples = [paddedProbs] #Use point estimates as first sample
                for i in range(dropout_samples):
                    modelPackage['model'].train()

                    packedOutput = modelPackage['model'](packedChars)

                    packedProbs = PackedSequence(F.sigmoid(packedOutput.data),packedOutput.batch_sizes)
                    paddedProbs,lengths = torch.nn.utils.rnn.pad_packed_sequence(packedProbs)

                    samples.append(paddedProbs)

                stds = torch.cat(samples,dim=2).std(dim=2)

                if return_char_scores:
                    batchDF['dropout_sds'] = [x[:l].ravel() for x,l in zip(stds.t().cpu().numpy(),lengths)]

                batchDF['dropout_sd'] = stds.sum(dim=0).cpu().numpy()

        if not return_char_scores:
            batchDF = batchDF.drop(['probs'],axis=1)

        batchResults.append(batchDF)

    allBatchesDF = pd.concat(batchResults)
    resultsDF = pd.merge_ordered(resultsDF,allBatchesDF,on='string')

    return resultsDF


def charScoreArray(chars,scores,max_len=1000):
    width = min(max(len(p) for p in scores),max_len)
    scoreArray = np.zeros((len(scores),width))
    for i,p in enumerate(scores):
        p = p[:width]
        scoreArray[i,:len(p)] = p

    charArray = np.vstack([np.array(list(c.decode('ascii')[:width])+['']*(width - len(c))) for c in chars])
    return charArray,scoreArray


def transparent_cmap(cmap, N=255):
    '''Copy colormap and set alpha values
    from https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image'''

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap



def plotCharProbs(chars,probs,uncertainties=None,max_len=1000,n_colors=4,gamma=1.5):
    charArray,probArray = charScoreArray(chars,probs,max_len)

    plt.figure(figsize=(0.1*charArray.shape[1],0.23*len(chars)))

    palette = sb.cubehelix_palette(light=1,dark=0.6,hue=1,start=1,rot=1,n_colors=2,gamma=gamma)
    sb.heatmap(probArray,annot=charArray,cmap=palette,fmt='',xticklabels=False,yticklabels=False,annot_kws={'family':'monospace'},cbar=False)

    if uncertainties is not None:
        palette = sb.cubehelix_palette(light=1,dark=0.6,hue=2,start=1,rot=1,n_colors=2,gamma=gamma)
        sb.heatmap(probArray,annot=charArray,cmap=palette,fmt='',xticklabels=False,yticklabels=False,annot_kws={'family':'monospace'},cbar=False)

        charArray,uncertaintyArray = charScoreArray(chars,uncertainties,max_len)
        cmap = sb.cubehelix_palette(light=1,dark=0.6,hue=3,start=0.7,rot=0,gamma=gamma,as_cmap=True)
        cmap = transparent_cmap(cmap)
        sb.heatmap(uncertaintyArray,cmap=cmap,fmt='',xticklabels=False,yticklabels=False,annot_kws={'family':'monospace'},cbar=False)
    else:
        palette = sb.cubehelix_palette(light=1,dark=0.6,hue=2,start=1,rot=1,n_colors=n_colors,gamma=gamma)
        sb.heatmap(probArray,annot=charArray,cmap=palette,fmt='',xticklabels=False,yticklabels=False,annot_kws={'family':'monospace'},cbar=False)


def scoreTestStringsDF(testStringsDF,modelPackage):
    matchesDF = extract(testStringsDF['string'],modelPackage,return_char_scores=True)#,hash_function=hash_function)

    testDataDF = buildTrainingData(testStringsDF,max_len=1000000)

    matchesDF = pd.merge(matchesDF,testDataDF,how='left',on='chars')
    matchesDF['probs'] = matchesDF['probs'].apply(torch.from_numpy)

    matchesDF['loss'] = [float(F.binary_cross_entropy_with_logits(probs,label,size_average=False)) for i,probs,label in matchesDF[['probs','labels']].itertuples()]
    matchesDF['accuracy'] = [float(((probs>0.5)==(label>0.5)).float().mean()) for i,probs,label in matchesDF[['probs','labels']].itertuples()]
    matchesDF['correct'] = matchesDF['accuracy'] == 1

    matchesDF = matchesDF.drop(['probs','labels'],axis=1)

    return matchesDF


def corruptTrainingStrings(trainingStringsDF):
    corrupt = np.array([not all(stringToAscii(m) in stringToAscii(s) for m in matches) for i,s,matches in trainingStringsDF[['string','matches']].itertuples()])

    return corrupt



if __name__ == '__main__':

    # Test code----------------------------------------------------------------------

    trainingStringsDF = loadTrainingStrings(r'C:\Users\Brad\Google Drive\Research\regcomments\data\orgs\extraction\trainingData\regDotGovCommenterOrg_train.csv',encoding='utf8')
    testStringsDF = loadTrainingStrings(r'C:\Users\Brad\Google Drive\Research\regcomments\data\orgs\extraction\trainingData\regDotGovCommenterOrg_test.csv',encoding='mbcs')

    trainingStringsDF = trainingStringsDF.sample(100)

    #Initialize org extractor
    modelPackage = newModel(d=100,cuda=True,lr=1e-3)

    #Train model
    historyDF = trainModel(modelPackage,trainingStringsDF,testStringsDF=testStringsDF,epochs=10,minibatch_size=20,verbose=True)
    historyDF = trainModel(modelPackage,trainingStringsDF,epochs=10,minibatch_size=20,verbose=True)
    plotLossHistory(historyDF)

    saveModelPackage(modelPackage,r'C:\Users\Brad\Google Drive\Research\Python3\subex\trainedModels\test.bin')
    loadedModelPackage = loadModelPackage(r'C:\Users\Brad\Google Drive\Research\Python3\subex\trainedModels\test.bin')

    historyDF = trainModel(loadedModelPackage,trainingStringsDF,epochs=3,minibatch_size=(5,20),verbose=True)
    plotLossHistory(historyDF)

    resultsDF = extract(trainingStringsDF['string'].sample(10),modelPackage,return_char_scores=True,dropout_samples=3)

    plotCharProbs(resultsDF['chars'],resultsDF['probs'])
    plotCharProbs(resultsDF['chars'],resultsDF['probs'],resultsDF['dropout_sds'])
