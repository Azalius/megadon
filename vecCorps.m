realNews = readtable('True.xlsx', 'range', 'A:B');
fakeNews = readtable('Fake.xlsx', 'range', 'A:B');
sizeFakeNews=int16(size(fakeNews, 1));
sizeRealNews=int16(size(realNews, 1));

realNews = [realNews array2table(ones(sizeRealNews, 1))]; 
%fill with either O or 1 (if its fake or not)
fakeNews = [fakeNews array2table(zeros(sizeFakeNews, 1))];

fakeNews = fakeNews(randperm(sizeFakeNews), :); 
%random permutation, shuffle all rows to be sure test & verif dataset 
%are picked from random place in the file
realNews = realNews(randperm(sizeRealNews), :);

news = [realNews ; fakeNews];
news.Var1 = categorical(news.Var1);
cvp = cvpartition(news.Var1,'Holdout',0.1);
dataTrain = news(cvp.training,:);
dataTest = news(cvp.test,:);

textDataTrain = dataTrain.text;
textDataTest = dataTest.text;
YTrain = dataTrain.Var1;
YTest = dataTest.Var1;

documents = preprocessText(textDataTrain);

bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];

XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear');

documentsTest = preprocessText(textDataTest);
XTest = encode(bag,documentsTest);

YPred = predict(mdl,XTest);

%on cherche les erreurs de prediction
fnText = textDataTest(find((YPred~=YTest) + (YPred==categorical(0)) == 2));
fpText = textDataTest(find((YPred~=YTest) + (YPred==categorical(1)) == 2));

acc = sum(YPred == YTest)/numel(YTest); %exactitude

%nombre faux positifs, faux negatifs, etc
TN = sum((YPred==YTest) + (YPred==categorical(0)) == 2); 
FN = sum((YPred~=YTest) + (YPred==categorical(0)) == 2);
FP = sum((YPred~=YTest) + (YPred==categorical(1)) == 2);
TP = sum((YPred==YTest) + (YPred==categorical(1)) == 2);

