realNews = readtable('True.xlsx', 'range', 'A:C');
fakeNews = readtable('Fake.xlsx', 'range', 'A:C');
sizeFakeNews=int16(size(fakeNews, 1));
sizeRealNews=int16(size(realNews, 1));

realNews = [realNews array2table(ones(sizeRealNews, 1))]; %fill with either O or 1 if its fake or not
fakeNews = [fakeNews array2table(zeros(sizeFakeNews, 1))];
fakeNews = fakeNews(randperm(sizeFakeNews), :); %random permutation, shuffle all rows to be sure test & verif dataset arent different
realNews = realNews(randperm(sizeRealNews), :);

news = [realNews ; fakeNews];
news.Var1 = categorical(news.Var1);
cvp = cvpartition(news.Var1,'Holdout',0.1);
dataTrain = news(cvp.training,:);
dataTest = news(cvp.test,:);

textDataTrain = dataTrain.ttl;
textDataTest =  dataTest.ttl;
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


XTest = encode(bag,documentsTest); %we want the test data, as a bag of words
YPred = predict(mdl,XTest); %we do the prediction
acc = sum(YPred == YTest)/numel(YTest) 
%we calc the accuracy (right prediction over total prediction)

