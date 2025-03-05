# gptopic

- [x] Create venv and install TF and required libraries
- [ ] Register it in jupyter notebooks see [link](https://stackoverflow.com/questions/42449814/running-jupyter-notebook-in-a-virtualenv-installed-sklearn-module-not-available) for more details.
    * pip install jupyter
    * python -m ipykernel install --user --name=uol_project
- [ ] Pull data
    - [ ] Oversample negative comments since positive ones are not very useful for detecting topics (i.e lots good, great, and that is it.)
- [ ] Label data with Gemini
- [ ] Manual QA of Labeling
- [ ] Train a sequential neural net to identify topics
    - [ ] Use functional API to identify topics (e.g Billing) and attributes (e.g horrible)
- [ ] Save pretrained model
- [ ] Run pretrained on a new data set
- [ ] Create a Human in the loop process
- [ ] Package model
- [ ] Deploy Django app