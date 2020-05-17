# DataMiningProj2_2020

TO-DOS

1. ~~Δημιουργία ενός αρχείου tsv από όλα τα txt του fulltext.zip το οποίο θα έχει τις στήλες Id, Title, Content και Category~~
1. ~~Split του παραπάνω αρχείου σε 2 tsv αρχεία train_set και test_set όπου το test_set να μην έχει το πεδίο Category. Με χρήση της συνάρτησης train_test_split και χρήση stratify parameter.~~
1. ~~Δημιουργία 5 wordClouds από την στήλη content. 1 για κάθε κατηγορία.~~
1. ~~Classification με:~~
    - ~~SVM~~ και πειραματισμός με παραμέτρους εκφώνησης
    - ~~Random Forests~~
    - ~~Naive Bayes~~
    - ~~KNN δική μας υλοποίηση~~ (λένε στο e-class ότι θέλουν και από το 10 CV + Roc Plot)
1. ~~Vectorization με:~~
    - ~~BoW (CountVectorizer)~~
    - ~~TfIdf~~
1. ~~Αξιολόγηση:~~
    - ~~Precision / Recall / F-Measure~~
    - ~~Accuracy~~
    - ~~ROC plot~~ ~~(λένε στο e-class ότι θέλουν και από το 10 CV)~~
1. Beat the Benchmark:
    - ~~Αφαίρεση stop words~~
    - ~~Stemming~~
    - Χρήση του τίτλου
1. Clustering
    - ~~Kmeans (μπορούμε και από κάποια υλοποιήση στο ιντερνετ ή βιβλιοθηκη, check eclass)~~
    - ~~Χρήση Cosine Similarity~~
    - ~~Θα εφαρμοστεί στο training set~~
    - ~~Χωρίς χρήση μεταβλητής Category~~
    - ~~Να γίνει με:~~
        - ~~Bow (CountVectorizer)~~
        - ~~TfIdf~~
        - ~~Pre-trained word embedings ένα από τα 3 προτεινόμενα της εκφώνησης~~
    - Οπτικοποίηση μέ:
        - ~~PCA~~
        - SVD
        - ICA