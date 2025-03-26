# gptopic

- [x] Create venv and install TF and required libraries
- [x] Register it in jupyter notebooks see [link](https://stackoverflow.com/questions/42449814/running-jupyter-notebook-in-a-virtualenv-installed-sklearn-module-not-available) for more details.
    * pip install jupyter
    * python -m ipykernel install --user --name=uol_project
- [x] Pull data
    - [x] Oversample negative comments since positive ones are not very useful for detecting topics (i.e lots good, great, and that is it.)
- [x] Label data with Gemini
- [x] Manual QA of Labeling
- [x] Train a sequential neural net to identify topicstrained model
- [ ] Create preprocessing pipelines. Modularize code
- [x] Run pretrained on a new data set
    - 
    - [ ] Generate tags (groun truth)
    - [ ] Show degrading accuracy of pretrained NN
    - [ ] Use functional API to identify topics (e.g Billing) and attributes (e.g horrible) and retrain last layer using TL
- [ ] Save as new model version (bigger model)
- [ ] Create a Human in the loop process (??)
- [ ] Package model
- [ ] Deploy Django app