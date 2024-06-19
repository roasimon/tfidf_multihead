# Text classification

This project consist in testing a multi-head architecture comparing to a single head text classification for predicting the sex author of a given text.\
We want to use date information because we assume that writing style depend on the author sex but also on publication date.\
To ease date classification we group date in 25 yeaes interval.\
Below are the models architecture used for this experience.\
<img src="/img/models.jpeg">
\
The features used are TF-IDF vectors extracted from the whole text corpus.\
<img title="High level view of method" src="/img/high_level_view.jpeg">
\

For the multi-head architecture, the total loss is the sum of sex and date interval losses.\
Below the first results:\
<img title="Performances of both models" src="/img/first_results.jpeg">
\
Those results are not satisfying on multi-head because date loss is superior to sex loss (date: 8 classes, sex: 2 classes).\
<img src="/img/loss_date.jpeg">
\
Thus we need to weight date loss.\
Below are results with weighted date loss:\
<img src="/img/date_weights_results.jpeg">
\
Performances are better for sex classification, but not that much for date classification.\
For this issue, we compute a new date loss based on date interval predictions and real date.\
First we multiply each date interval probability to their class, and sum up those to have a new date prediction.\
Then we compare that new prediction to the real date using Mean Squared Error.\
This will be our new date loss, but as showed below it is needed to be weighted:\
<img src="/img/new_date_loss.jpeg">
\
Below are results with our new weighted date loss:\
<img src="/img/new_date_loss_results.jpeg">
\
Great to know that the sex classification is better with new date loss, but performances on date MAE is not better.\
But the hypothesis is still confirmed with below results:\
<img src="/img/final_results.jpeg">
