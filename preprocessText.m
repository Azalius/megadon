function documents = preprocessText(textData)
    %permet de preprocess un ou des documents 
  
    documents = tokenizedDocument(textData);

    documents = addPartOfSpeechDetails(documents);
    documents = removeStopWords(documents);
    documents = normalizeWords(documents,'Style','lemma');
    %on enleve les stops word & on lemmatize le texte

    % On effece la poctuation
    documents = erasePunctuation(documents);

    %Et les mots trop courps ou trop long
    documents = removeShortWords(documents,2);
    documents = removeLongWords(documents,15);

end