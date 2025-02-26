# SMI_CM

## Project description
This project focuses on developing a few-shot classifier trained on transcribed interviews from a job center in Denmark. The classifier is designed to perform binary classification of reported speech. This repository provides a two-part pipeline that utilizes a trained model to process interview documents and highlight key sentences. The pipeline performs the following tasks:

:::info
Input Document Handling: Accepts a .docx document as input, containing the interview transcript.

Sentence Segmentation: Breaks down the input document into individual sentences.

Sentence-Classification: Applies the trained model to make predictions on each sentence, classifying them based on the provided criteria.

HTML-Based Highlighting: Highlights the classified sentences using HTML tags for visual representation in the outputfile.:star: 

Output Generation: Produces a .docx document with highlighted sentences, preserving the original content while adding visual indicators for classified segments.

A Gooey-program wrapper is made which wraps the the above functions into a .exe program.

All scripts are written in python:snake:
:::

### Main Python Packages:
Transformers: For SetFit framework, tokenizer and overall workflow with Language Modelling.
PyTorch: For training schedules and GPU-support.
SetFit: For Building the Few-shot Model.
scikit-learn: For K-Fold validation.
Gooey: For creating simple user interfaces.
Pysbd: For splitting text into sentences.
spire.doc: For reading and writing .docx files.

An environment.yml file is included to recreate the setup needed for running all scripts.:package: 



# Repo parts 
The repo is separated in 3 parts:

- Modelling
- Setup file for freezing it to .exe file
- Highlighter with Gooey-wrapper

The modelling part is split into five python scripts which handles necessary parts of data preparation, model training and evaluation of the SetFit-Model.



## Few-Shot Learning
Few-Shot Learning (FSL) is a machine learning method which can be classified from Wang et. al. 2022 as: 

> A computer proram is said to learn from experience E with respect to some classes of task T and performance measure P if its performance can improve with E on T measured by P.

E in FSL is typically very small and in vein of this project also quite small with only 25 texts in total.

The Task at hand here is classifying reported speech in danish interviews with binary labels ("reported speech", "not reported speech")
Few-shot learning (FSL) is popular for its ability to learn and predict classes and instances with only a few training samples, making it particularly useful in scenarios where data is sparse and traditional language model solutions are not feasible.

The model is based on a pretrained LLM with a logistic regression as prediction head.

> FSL in the larger context of machine learning called N-way-K-shot. It represents N-categories and K-samples.

## Modelling details

The modelling contains five python scripts:

### 1: Data preparation:
The first script in this modelling part is responsible for preparing the dataset used to train the model on reported speech sentences. As mentioned above the sentences are annotated from earlier interviews conducted by the reseacher and then annotated.
The script Loads data from a json lines format, splits it into sentences, excludes parts with no sentences and saves as .json file for easier handling as input for training the model.
The handling of data is done using the function `get_sentences` from the `smi_sm_funs.py` script which uses the `pysbd` package to split text into sentences, stores the labels into the labels list, creates label index pairs and then splits the sentences into negatives and positives (reported-speech/not-reported-speech) labelled at the start 0 and 1.


### 2: Hyperparameter optimization:
The project utilizes a SetFit model because of the small sample of texts which consists of 55 in total. A random set of 25 of these are then chosen which are split into train, test and evaluation sets.

The model uses a simple binary classification of reported speech/not reported speech.

The data is split into a 70/30 split for training and testing. The 30% for testing are further split into 15% for evaluation and 15% for testing.

During data preparation, it was ensured that the data set contained an equal number of samples for each label.

The model is build upon the [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) model from :hugging_face:. The pretrained model was chosen because of its placement on the :hugging_face: model leaderboard for Danish at the time of training.


:::success
Exploring other pretrained models for fine-tuning is not only welcomed but also encouraged to potentially enhance the final model's performance.
:::

The script uses the normal training setup from the `SetFit` library and uses the built-in functions for hyperparameter-search.
15 trials were run with different training parameters.

Best performance was achieved with the following hyperparameters which were used for training the final model.
```
BestRun(run_id='11', objective=3.9052886043810604, hyperparameters={'body_learning_rate': 1.0770502781075495e-06, 'num_epochs': 6, 'batch_size': 32, 'max_iter': 279, 'solver': 'lbfgs'}, backend=<optuna.study.study.Study object at 0x7f7bbb289490>)
```


### 3: Cross-validation

Next script utilizes a Stratified K-fold cross-validation scheme for validating the reliability of the model's performance.
Five folds were chosen in this use case and the best logged parameters from the hyperparameter-search was used.

The results from the cross-validation is as follows:
``` 
[{
    "fold": "average",
    "accuracy": 0.9688073394495411,
    "precision": 0.9725369355275898,
    "recall": 0.965137614678899,
    "f1": 0.9687284674485521
},
{
    "fold": 5,
    "accuracy": 0.981651376146789,
    "precision": 0.9906542056074766,
    "recall": 0.9724770642201835,
    "f1": 0.9814814814814815
},
{
    "fold": 4,
    "accuracy": 0.9724770642201835,
    "precision": 0.9904761904761905,
    "recall": 0.9541284403669725,
    "f1": 0.9719626168224299
},
{
    "fold": 3,
    "accuracy": 0.963302752293578,
    "precision": 0.954954954954955,
    "recall": 0.9724770642201835,
    "f1": 0.9636363636363636
},
{
    "fold": 2,
    "accuracy": 0.9678899082568807,
    "precision": 0.9636363636363636,
    "recall": 0.9724770642201835,
    "f1": 0.9680365296803652
},
{
    "fold": 1,
    "accuracy": 0.9587155963302753,
    "precision": 0.9629629629629629,
    "recall": 0.9541284403669725,
    "f1": 0.9585253456221198
}]
```
The results suggests the accuracy was not the result of random selection of the train/test split and that the model succesfully learned to classify results. The average fold's accuracy ends up being ==96.88== quite close to the lowest folds accuracy of 95.87% which again suggests that the models training data and hyperparameters resulted in a good performing model.

### 4: Model training
The model is trained based on results from the hyperparameter-search results and results in an accuracy of ==~96%==. This demonstrates strong performance in classifying reported speech, even with the limited texts from the train, test, and evaluation sets. Training was conducted on a High-Performance Computing (HPC) system, utilizing a single NVIDIA H100 192 GB GPU.

The parameters for the final training schedule is shown below utilizing parameters from HPO:

```
### Load model (using text labels)
model = SetFitModel.from_pretrained(
    model_name, 
    labels=["not reported speech", "reported speech"],
    ### Hyperparameters (best run from HPO)
    head_params={
            'max_iter': 300, 
            'solver': 'lbfgs'
        }
).to(device)

# Initialiser træningsargumenter
args = TrainingArguments(
    ### Hyperparameters (best run from HPO - see hpo/hpo_bestrun.txt)
    batch_size=32,
    num_epochs=6,
    evaluation_strategy="epoch",
    body_learning_rate = 1.0770502781075495e-06,
    save_strategy="epoch",
    load_best_model_at_end=True
)
```
### 5: Model evaluation

Performance metrics for final model can be seen in the table below.

|          | Not Reported Speech | Reported Speech|
| -------- | -------- | -------- |
|Precision | 0.959     | 0.927   |
|Recall    |0.924     | 0.961
|F1 | 0.941|0.943
|Accuracy| | 0.942


We evaluated how the model performance varied across varying probability threshold for labelling a sentence as "reported speech". We evaluated between a threshold of 0.5 (the default) and 0.9 in increments of 0.05. This process assesses whether the model's overall performance improves with higher probability thresholds, effectively filtering out results where the model is less confident in its predictions. 
A training loop is written in the `05_model_eval.py` script to test this.

The results from the model evaluation where the incremental increase of the threshhold was utilizied is show below:
```
[{"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.5}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.55}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.6}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.65}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.7}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.75}, {"not reported speech": {"precision": 0.9591549295774648, "recall": 0.9240162822252375, "f1-score": 0.9412577747062889, "support": 737.0}, "reported speech": {"precision": 0.9267015706806283, "recall": 0.9606512890094979, "f1-score": 0.9433710859427049, "support": 737.0}, "accuracy": 0.9423337856173677, "macro avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244969, "support": 1474.0}, "weighted avg": {"precision": 0.9429282501290466, "recall": 0.9423337856173677, "f1-score": 0.9423144303244968, "support": 1474.0}, "threshold": 0.8}, {"not reported speech": {"precision": 0.9592123769338959, "recall": 0.9253731343283582, "f1-score": 0.9419889502762431, "support": 737.0}, "reported speech": {"precision": 0.927916120576671, "recall": 0.9606512890094979, "f1-score": 0.944, "support": 737.0}, "accuracy": 0.9430122116689281, "macro avg": {"precision": 0.9435642487552836, "recall": 0.9430122116689281, "f1-score": 0.9429944751381215, "support": 1474.0}, "weighted avg": {"precision": 0.9435642487552836, "recall": 0.9430122116689281, "f1-score": 0.9429944751381215, "support": 1474.0}, "threshold": 0.85}]
```

As performance does not vary noticable, we conclude that maintaining the default threshold of 0.5 works well for this model.

### Methodological considerations

The amount of texts used in training was considered heavily in the process. As Few-Shot Learning was used to predict labels, it was needed to access how big the corpus should be in order to get as good a score as possible and to not overfit the training data, as FSL-models has a tendency to do. The choice landed on a random sample of 25 texts which as mentioned was split into train, test and eval sets. The Few-Shot learning methods was ultimately chosen because of the inherently few available texts. A method that could effectively predict the desired labels in sparse data was needed. Experiments were conducted to determine the optimal size of the corpus. A key challenge during data preparation was not only deciding the corpus size but also ensuring an appropriate distribution of texts. This was necessary for the model to learn the distribution of both the texts and their corresponding labels. To ensure the model learned a representative amount of label variations and avoided overfitting on one label over another, a strict balance of labels was maintained in the training data. This balance was achieved manually by the researcher who labelled the training.
In the litterature in the area of applied Few-Shot Learning, the optimal amount of text is also discussed and its problems such as overfitting, transfer learning, data augmentation and how to account for them. ==A good overview of FSL can be found in Song et. al. 2022.==


## Highlighter
:::info
This Part goes into detail of the highlighter functions and the Gooey wrapper.
:::
The highlighter uses the spire.doc package to read and write the input/output file as .docx and further converting the original document to HTML.
The highlighter uses functions from the `SMI-funs.py` script. These functions are called by the highlighter inside the `process_text` pipeline.

### Gooey wrapper
The Gooey wrapper is used to wrap the PyInstaller-generated .exe file, creating a simple graphical user interface (GUI) for the program. This approach ensures that users can locally install and utilize the analyzer program without requiring any prior Python experience.
Gooey essentially works as arg-parser where one creates Gooey-objects and uses Gooey-parser to parse these to the final program, i.e wrapping the objects such as  buttons, menus, drop-downs or documentations into working things GUI-wise inside the main Gooey function.


The finalised program:

![image](https://hackmd-prod-images.s3-ap-northeast-1.amazonaws.com/uploads/upload_466d29764983009c234ebc8a9805a9e4.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1740596597&Signature=8MkFSA26Z8qBMqO4ZvZHazn%2Fioo%3D)

A simple yet effective setup for this context that incorporates model, highlighter and saves as output. Huzzah!

For references and guidance visit the [Gooey](https://github.com/chriskiehl/Gooey) github page

## PyInstaller
:::info
This last part details the PyInstaller process, potential pitfalls and its workings with Gooey and installations of the program.
:::

Pysintaller is a python package to 'freeze' your script and turn it into a executable application.
The program was compiled using PyInstallers .spec file option which contained the wanted options. The .spec file was chosen for reproducability and to quickly compile a new version when correcting the script when bug fixing.

A --onefile option was chosen so as the PyInstaller compiles everything into one file that runs the script in one sitting. Also necessary so as to make it into a Gooey program.

When the program loaded the reported speech model, a native input/output issue arose during compilation. This was due to the use of sys.stdin, sys.stdout, and sys.stderr where the system could not load from memory and thus returning 'None'.

The PyInstaller page explains it as:
> A direct consequence of building your frozen application in the windowed/no-console mode is that standard input/output file objects, sys.stdin, sys.stdout, and sys.stderr are unavailable, and are set to None. The same would happen if you ran your unfrozen code using the pythonw.exe interpreter, as documented under sys.__stderr__ in Python standard library documentation.

At this project it relates to the loading of the fine-tuned model and it was solved with standard solution offered by PyInstaller documentation. It is included here as it was a problem that took awhile to figure out.

The solution:
```
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
```
Further dependencies was used as required in order for the program to compile it correctly. The hook-files was written in order to collect python-package dependencies for SetFit package and Gooey. Both written as python scripts. These handles the loading of the package dependencies for the libraries and makes sure the functions used works in the final application.

More binaries are compiled in the .spec file from the spire.doc package which was also needed in order for the application to convert and reconvert into word documents.

## Concluding remarks

This repository documents the process of building a text classifier for research purposes using a Few-Shot Learning (FSL) model. FSL was selected due to its effectiveness in handling small training data samples.

The final model is pushed to the CALDISS org. page on the [CALDISS-AAU](https://huggingface.co/CALDISS-AAU):hugging_face:-page where it can be downloaded and of course used for reported-speech sentence-classification. 

#### Future Directions

As Few-Shot Learning, Large Language Models (LLMs), and Natural Language Processing (NLP) continue to evolve, developing more mature frameworks and models for similar projects is highly encouraged.

We hope this repository serves as:

Inspiration: A resource for working with Few-Shot learning and language model classification in research and beyond.

Practical Guidance: A template for building simple GUI frameworks to support classification and Language models and for utilizing high-level Language models in a simple framework where deployment is at focus.

## References
* Song, Yisheng, Ting Wang, Subrota K. Mondal, and Jyoti Prakash Sahoo. ‘A Comprehensive Survey of Few-Shot Learning: Evolution, Applications, Challenges, and Opportunities’. arXiv, 24 May 2022. http://arxiv.org/abs/2205.06743.

* Yaqing Wang, Quanming Yao, James T Kwok, and Lionel M Ni. Generalizing from a few examples: A survey on few-shot learning. ACM Computing Surveys (CSUR), 53(3):1–34, 2020.

* E-iceblue. (n.d.). Spire.Doc for Python (Version [latest version]). Python Package Index. Retrieved September, 2024, from https://pypi.org/project/Spire.Doc/

* Kiehl, C. (n.d.). Gooey: Turn (almost) any Python command line program into a full GUI application with one line [Software]. GitHub. Retrieved September, 2024, from https://github.com/chriskiehl/Gooey

* PyInstaller Development Team. (2024). PyInstaller Manual (Version 6.11.1). Retrieved from https://pyinstaller.org/en/stable/index.html
