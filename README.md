# RecSys Challenge 2018

## Dataset generation
* [prepare_data.py](prepare_data.py) generates the binary user-item matrix from the downloaded dataset.
* [histograms.py](histograms.py) gives histograms of playlist and song statistics.

## Methods:
### Matrix factorization

* [nmf_cv.py](nmf_cv.py) runs cross-validation to determine the best hyperparameters. The hyperparameters searched for are:
  * Number of dimensions of latent space
  * Regularization constant
  * Weight for elastic net


### User-similarity based

* The file [Similarity_CF_user.py](Similarity_CF_user.py) runs user-based similarity for a specific set of hyperparameters.
* The file [Similarity_CF_user_cv.py](Similarity_CF_user_cv.py) runs cross-validation on the hyperaparameters before selecting the best one. The hyperparameters searched for are:
  * alpha: Parameter to tune similarity measure
  * q: Locality of scoring function

### Item-similarity based

* The file [Similarity_CF_item.py](Similarity_CF_item.py) runs item-based similarity for a specific set of hyperparameters.
* The file [Similarity_CF_item_cv.py](Similarity_CF_item_cv.py) runs cross-validation on the hyperaparameters before selecting the best one. The hyperparameters searched for are:
  * alpha: Parameter to tune similarity measure
  * q: Locality of scoring function

### Ensemble of user-similarity based and item-similarity based
* [Similarity_ensemble.py](Similarity_ensemble.py) uses a weighted average of the scores generated using the aforementioned methods to generate recommendations.
