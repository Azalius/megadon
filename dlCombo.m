realNews = readtable('True.xlsx', 'range', 'A:C');
fakeNews = readtable('Fake.xlsx', 'range', 'A:C');
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

textDataTrain = dataTrain.ttl;
textDataValidation = dataValidation.ttl;
YTrain = dataTrain.Var1;
YValidation = dataValidation.Var1;

documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);
enc = wordEncoding(documentsTrain);

documentLengths = doclength(documentsTrain);
% figure
% histogram(documentLengths)
% title("Document Lengths")
% xlabel("Length")
% ylabel("Number of Documents")

sequenceLength = 520; %rarements plus de 520 mots
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

