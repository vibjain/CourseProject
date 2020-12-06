# CS410 Text Information Systems
Generating a table of contents for a video transcript

Nitish Jain (nitishj2@illinois.edu)
Vibhor Jain (vibhorj2@illinois.edu)

Instructions to run:

1. Install python 3.8
2. Create Azure Subscription and Create Azure Cognitive Services resource, replace the key in secrets.py and the location 
of this resource in flaskwebpage.py (line 11), we used westus
3. Install dependent packages
    import numpy as np
    import pandas as pd
    import nltk
    import re
    import networkx as nx
    import os, pickle, re
    import sklearn
    import secrets, script
    import os, sys, json
    import azure.cognitiveservices.speech as speechsdk
    import moviepy.editor as mp
    import time

4. Run following commands python terminal before running the project
    python -m pip install azure-cognitiveservices-speech
    python -m nltk.downloader stopwords
    python -m nltk.downloader punkt

Sample video (Input)
https://www.ted.com/talks/severn_cullis_suzuki_make_your_actions_on_climate_reflect_your_words#t-212806

Sample TOC (Output)
00:12: world | speak | go |
04:08: momentumparis | agreement | limit |
04:15: times | climate | time |
05:19: say | reminded | science |
05:53: make | actions | reflect |

Glove Reference
https://nlp.stanford.edu/projects/glove/