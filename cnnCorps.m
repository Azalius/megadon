realNews = readtable('True.xlsx', 'range', 'A:B');
fakeNews = readtable('Fake.xlsx', 'range', 'A:B');
sizeFakeNews=int16(size(fakeNews, 1));
sizeRealNews=int16(size(realNews, 1));

realNews = [realNews array2table(ones(sizeRealNews, 2))]; %fill with either O or 1 if its fake or not
fakeNews = [fakeNews array2table(zeros(sizeFakeNews, 2))];
fakeNews = fakeNews(randperm(sizeFakeNews), :); %random permutation, shuffle all rows to be sure test & verif dataset arent different
realNews = realNews(randperm(sizeRealNews), :);

news = [realNews ; fakeNews];

news.Var1 = categorical(news.Var1);
cvp = cvpartition(news.Var1,'Holdout',0.1);
dataTrain = news(cvp.training,:);
dataValidation = news(cvp.test,:);

textDataTrain = dataTrain.text;
textDataValidation = dataValidation.text;
YTrain = dataTrain.Var1;
YValidation = dataValidation.Var1;

documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);
enc = wordEncoding(documentsTrain);

sequenceLength = 500;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);

XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);


inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

function documents = preprocessText(textData)

    % Tokenize the text.
    documents = tokenizedDocument(textData);

    % Remove a list of stop words then lemmatize the words. To improve
    % lemmatization, first use addPartOfSpeechDetails.
    documents = addPartOfSpeechDetails(documents);
    documents = removeStopWords(documents);
    documents = normalizeWords(documents,'Style','lemma');

    % Erase punctuation.
    documents = erasePunctuation(documents);

    % Remove words with 2 or fewer characters, and words with 15 or more
    % characters.
    documents = removeShortWords(documents,2);
    documents = removeLongWords(documents,15);

end

