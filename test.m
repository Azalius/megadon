%fichier permettant de classifier des données hors du jeu d'essai.
%Il d'abord entrainer un modele pour pouvoir le tester ic


str = [ %un exemple extrait du net
    "How times can change. A few months ago, North Korea’s dictator, Kim Jong-un, was firing missiles over Japan and threatening to send nuclear bombs in our direction. Now North Korea has agreed to open its doors to food, medicine and the message of Christ.But last week, the young North Korea leader dropped a different kind of surprise on the world: He met with South Korean president Moon Jae-in on April 27 and announced that the 67-year-old Korean conflict is over.  I came here to put an end to the history of confrontation, Kim Jung-un told Moon in a meeting on the border town of Panmunjom.There will be no more war on the Korean peninsula, and a new age of peace has begun,  the two leaders said in a joint statement.Charisma Magazine reports: Kim Jong-un, who has built the fourth largest army in the world—with 1.19 million soldiers—says he will now focus on rebuilding his country’s shattered economy.Boom. Just like that, swords were converted into plowshares. The two leaders, all smiles for the cameras, agreed they will denuclearize the Korean peninsula within a year. They also agreed to set up reunions with families that have been divided since the Korean War started in 1950.It feels like we should declare a global holiday and dance in the streets. But most Americans were too distracted by the opening of the new Avengers movie to pay attention to the headlines.What was behind the Korean surprise? Most media outlets didn’t notice that Christians in South Korea had been fasting and praying for the peace summit. Pastors held an all-night vigil in the city of Paju, south of the North Korean border. And a group of Christian politicians held a fasting and prayer event in the National Assembly buildings in Seoul, according to Yonhap News.North Korea’s persecuted Christians have also been praying for this moment—for years. They have been horribly persecuted. They have been forced to meet secretly. They have been routinely rounded up and sent to labor camps—or just shot on sight—because they did not worship Kim Jong-un as their god."
  ]
documentsNew = preprocessText(str);

%XNew = doc2sequence(enc,documentsNew,'Length',17); %pour le dl
%labelsNew = classify(net,XNew) 

XNew = encode(bag,documentsNew); % pour la SVM
labelsNew = predict(mdl,XNew)











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