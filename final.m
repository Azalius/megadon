clear all;
clc;
T = readtable('cancer.xls');
X = T(:,1:30);
Y = T(:,31);
N = size(X,1);

N_2 = floor(N/2);
indices_tr = randi(N, N_2, 1);

X_tr = X(indices_tr,:);
X_ts = X(indices_tr,:);
Y_tr = Y(indices_tr,:);
Y_ts = Y(indices_tr,:);

X_ar =  table2array(X(indices_tr,:))';
Y_ar =  table2array(Y(indices_tr,:))';


%On cree le classificateur de Bayes
mod_bayes = fitcnb(X_tr, Y_tr);

%On cree le classificateur k plus proches voisin
mod_knn = fitcknn(X_tr, Y_tr);

%on cree le fit SVM
mod_svm = fitcsvm(X_tr, Y_tr);

%on fit la regression logistique
Y_ar2 = categorical(~Y_ar);
mod_rl = mnrfit(X_ar', Y_ar2');

%On cree et on entraine le reseau de neurone
net = feedforwardnet(10);
net = configure(net, X_ar, Y_ar);
[mod_rn, tr] = train(net, X_ar, Y_ar);

n=10;
%on debute la cross comparaison
indices_ts = setdiff([1:N]', indices_tr);
indices = crossvalind('Kfold', N, n);

%On initialise toutes les erreurs
erreur_bayes = 0;
erreur_rl= 0;
erreur_kppv = 0;
erreur_svm = 0;
erreur_rn = 0;

for i = 1:n
    
    % Extraire les donn�es d'entrainement
    X_tr = X(indices ~=  i,:);
    Y_tr = Y(indices ~=  i,:);
    N_tr = size(Y_tr,1);
    
    % Extraire les donnees de validation
    X_ts = table2array(X(indices == i,:));
    Y_ts = Y(indices == i,:);
    Y_ts2 = table2array(Y_ts);
    N_ts = size(Y_ts,1);

    X_tr2 = table2array(X_tr);
    Y_tr2 = categorical(table2array(Y_tr));

    %On fait la prediction de chaque modele
    pr_cn = mod_rn(X_ts'); 
    pr_bayes = mod_bayes.predict(X_ts);
    pr_rl = mnrval(mod_rl, X_ts);
    pr_kppv = mod_knn.predict(X_ts);
    pr_scm = mod_svm.predict(X_ts);

   e_bayes = 0.0;e_rl=0; e_kppv=0;e_scm=0; e_cn=0; %initilsie compteur derreur temporaire
   for j =1:N_ts; %pour chaque cas de test
       if(abs(pr_cn(j) - Y_ts2(j)) >= 0.5); %on test si le rn predit le bon chiffre, cad si il est a mons de 0.5 de la bonne valeur. 
         e_cn = e_cn+ 1.0;
       end
       if(pr_bayes(j) ~= Y_ts2(j)); %Si le predit est dfferent du reel, on rajoute une erreur 
         e_bayes = e_bayes+ 1.0;
       end
       if(pr_scm(j) ~= Y_ts2(j)); %Si le predit est dfferent du reel, on rajoute une erreur 
         e_scm = e_scm+ 1.0;
       end
       if(pr_kppv(j) ~= Y_ts2(j)); %Si le predit est dfferent du reel, on rajoute une erreur 
         e_kppv = e_kppv+ 1.0;
       end
       if(abs(pr_rl(j) - Y_ts2(j)) >= 0.5); %on test si la regression predit le bon chiffre, cad si il est a mons de 0.5 de la bonne valeur.
           e_rl = e_rl+ 1.0;
       end
   end
   e_bayes = e_bayes / N_ts;  e_rl=e_rl/ N_ts;e_kppv=e_kppv/N_ts;e_scm=e_scm/N_ts;e_cn=e_cn/N_ts;
   
   %On aditionne toutes les erreurs
   erreur_bayes =  erreur_bayes + e_bayes;
   erreur_rl=erreur_rl+e_rl;
   erreur_kppv = erreur_kppv+e_kppv;
   erreur_svm=erreur_svm+e_scm;
   erreur_rn=erreur_rn+e_cn;
end

%on fait la moyenne des erreurs
erreur_bayes =  erreur_bayes / n;
erreur_rl =  erreur_rl / n;
erreur_kppv =  erreur_kppv / n;
erreur_svm =  erreur_svm / n;
erreur_rn =  erreur_rn / n;

fprintf('Erreur de validation croisee Bayes : %f \n', erreur_bayes);
fprintf('Erreur de validation croisee regression logistique : %f \n', erreur_rl);
fprintf('Erreur de validation croisee KPPV : %f \n', erreur_kppv);
fprintf('Erreur de validation croisee SVM : %f \n', erreur_svm);
fprintf('Erreur de validation croisee reseau neuronnes :  %f \n', erreur_rn);

%La regression logistique et le reseau de neuronns sont les plus
%performants avec plus de 97% de precision. Le reseau de neuronne est le
%plus précis, surtout si on augmente le nombre de neuronnes. Tout les
%modéles ont plus de 92% de precision.