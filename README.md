# Reproducability Study on Multi-emotion detection in user-generated reviews

## Original Paper and Data source
L. Buitinck, J. van Amerongen, E. Tan and M. de Rijke (2015).
`Multi-emotion detection in user-generated reviews
<https://www.researchgate.net/publication/272677182_Multi-Emotion_Detection_in_User-Generated_Reviews/links/54eb26230cf2f7aa4d5a63d4.pdf>`_.
Proc. 37th European Conference on Information Retrieval (ECIR).

## Data source
https://github.com/NLeSC/spudisc-emotion-classification

## How to run

We use Docker to ensure that the experimetns are reproducable.
Debian Buster with Python Version 3.8.7 is used as the base image.

docker build -t reproducability_multi-emotion_detection:1.0 .

docker run -v {/absolute/path/to/repository}/output:/code/output/ reproducability_multi-emotion_detection:1.0