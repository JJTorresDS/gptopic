# GPTopic

GPTopic is a topic modeling framework that abstracts the nuances of Natural Language Processing and Deep Learning.

It provides a high level API that allows you to label "unlabeled" reviews using a Large Language Model and then train a deep neural network the, now, labeled data.

Benefits:
* Converts an unsupervised learning problem into a supervised learning problem.
* Abstracts the knowledge need to train neural networks specialized in text processing.

## Quick Start for Mac and Linux users
1. Clone this repository: git clone https://github.com/JJTorresDS/gptopic.git
2. On the terminal run the following commands to create a virtual environment:
    * python3 -m venv venv
    * source venv/bin/activate
3. On the terminal, and after activating the virtual environment run: pip install -r requirements.txt
4. On the terminal run the following comman to open a jupyter notebook: jupyter notebook
4. Create a [GEMINI API key (its free!)](https://aistudio.google.com/app/apikey)
5. Open the "Quick start.ipynb" notebook.

Have fun !!
