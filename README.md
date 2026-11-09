# Fake News Detection

Classifying news articles from the ISOT Fake News dataset as real or fake, from simple baselines to TF-IDF models
and a fine-tuned BERT model, with a close look at what the models actually learn.

## Results

Accuracy on a held-out test set of 7,716 articles, after removing source traces and duplicates (see below):

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|
| BERT (`bert-base-uncased`, fine-tuned) | **0.998** | 0.998 | 1.000 |
| Logistic regression (TF-IDF + style features) | 0.997 | 0.997 | 1.000 |
| Random forest (TF-IDF + style features) | 0.995 | 0.995 | 1.000 |
| Naive Bayes (word counts + style features) | 0.941 | 0.940 | 0.975 |
| Title length rule | 0.831 | 0.826 | 0.878 |
| Always predict the majority class | 0.549 | 0.355 | – |

## What the models really learn

The near-perfect scores mostly measure **which source an article comes from, not whether it is true.** All real
articles in ISOT come from Reuters and all fake ones from a few websites, and each has its own house style:

- The `subject` categories of real and fake articles do not overlap, so the subject alone gives 100 % accuracy.
- 99 % of real articles start with a "CITY (Reuters) -" dateline; fake ones carry image credits, links and
  Twitter handles from their websites; and thousands of texts are duplicates.
- These traces are removed before modelling, but a text-only model still drops only from 99.6 % to 98.7 %, and
  titles alone give 95.2 %. The strongest remaining features are style habits: Reuters writes "said", names the
  weekday and gives times in EDT, while the fake sites write headlines in Title Case and advertise their own shows.

A model trained on this data is therefore unlikely to work on fake news from other sources. An earlier version of
this project trained on the uncleaned data and reported 99.9 % accuracy; the notebooks now remove the shortcuts and
show how much they mattered.

## Notebooks

- [`EDA.ipynb`](EDA.ipynb): class balance, subjects, title and text lengths, source traces and duplicates.
- [`modeling_4.ipynb`](modeling_4.ipynb): cleaning, baselines, logistic regression, naive Bayes and random forest
  with grid search, the effect of the shortcuts, BERT, model comparison, error analysis and the most important
  words.

## Method

1. **Cleaning:** remove the Reuters dateline and publisher markers (URLs, Twitter links and handles, image credits,
   site names), drop empty and duplicate texts, and leave out `subject` and `date`. 38,579 of 44,898 articles remain.
2. **Split:** one stratified 80/20 train/test split; everything is fitted on the training set only.
3. **Features:** lemmatised titles and texts (TF-IDF or counts) plus style features: average word length,
   punctuation, share of capital letters and sentiment for the title and the text.
4. **Models:** logistic regression, naive Bayes and random forest tuned with 3-fold grid search; BERT fine-tuned for
   up to 2 epochs with early stopping on a validation split.

## Run it

1. Download the dataset from the
   [ISOT Fake News Dataset page](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
   (University of Victoria) and put `Fake.csv` and `True.csv` in a `data/` folder.
2. `pip install -r requirements.txt`
3. Run `EDA.ipynb`, then `modeling_4.ipynb`. The modelling notebook takes about 15 minutes with a GPU (BERT
   fine-tuning); without one, BERT is much slower.

## Next steps

- Test the models on a different fake-news dataset to measure how much of the accuracy is source style.
- Use data where real and fake articles come from similar outlets, or claim-level fact-checking data such as LIAR.
- Add temporal validation: train on older articles and test on newer ones.

## License

MIT Licence
