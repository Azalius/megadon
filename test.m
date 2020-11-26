realNews = readtable('True.xlsx', 'range', 'A:B');
fakeNews = readtable('Fake.xlsx', 'range', 'A:B');
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
acc = sum(YPred == YTest)/numel(YTest)

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