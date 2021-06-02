# Inferring Test Models using Multi-Objective Evolutionary Algorithms

This project aims at automatically inferring test models using bug reports.
After collecting the bug reports, processing them into execution traces, and then collecting a dictionary, a Multi-Objective Evolutionary Algorithm (MOEA) is used to find the best models.

With that in mind, this project is divided into 2 main functionalities:

1) Collect and process bug reports;
2) Evolve a set of models with MOEAs.

The next sections provide more information about each.

## Requirements

Our code uses Python 3.8.

The list of libraries needed for this project can be found in `libs.txt`.
You can install them in your virtual environment with `pip install -r libs.txt`.

## Collect and process bug reports

The content of this module can be found at `<project-dir>/src/NLP`.

The main script of this component is `TesterMain.py`, which will find the bugs in a given BugZilla repository and process them in the format needed in this project.
To run it, you can simply execute it as is.

The following lines are used to configure how it works:

```python
# Set the specific bugzilla website and (optional) authentication credentials
setting = {
    "url_jsonrpc": "https://bugzilla.mozilla.org/jsonrpc.cgi",
    "bugzilla_auth": {
        "login": "i",
        "password": "U"}
}
```

Change this piece of code to provide the URL and credentials for the BugZilla repository.
Bear in mind that the URL should point to the JSON-RPC GET service.

```python
flagProduct = 0
flagComponent = 0   # Search all component for a specific program, compulsory for the search based on all component
flagComputeSearch = 0     # Retrieve the bugs
flagJustParsing = 0       # Used to only test different parsing format of the comment text
flagTextCompClst = 1      # Compute the clustering phase
```

These are the flags used to define what are the steps you want to take.

1) `flagProduct` - set to 1 if you would like to find all program in the repository, otherwise change the following line to the specific program you want:
```python
if flagProduct:
    listProd = Bugzilla.findProd()
else:
    product = "krita"
```
In the example, we are looking for the program "krita".

2) `flagComponent` - set to 1 if you would like to find all components of the program, otherwise change the following line to the specific component you want:
```python
# If you want to search based on a specific component, decomment next line of code
component = "plus"
```
In the example, we are looking for the "plus" component.

3) `flagComputeSearch` - set to 1 if you want to retrieve and search again the bugs from the website, otherwise it will only perform part of the process (i.e., it will only parse the results already stored in disk).

**WARNING**: the BugZilla search seems to have changed. Hence, enabling this flag will cause an error because no bugs can be retrieved with our query code. This part must be updated in the future. All the other procedures work, such as language processing and clustering.

4) `flagJustParsing` - set to 1 if you would like to only parse the results.

5) `flagTextCompClst` - set to 1 if you would like to apply the clustering algorithm on the results (default).

After executing this script, the result will be added to `data/bug/dataset/BugStepsMappedPp_<program>_<component>.pkl`.
We already provide in this directory the dataset we have used in our experiments.
The other files are intermediate files before the data processing.

## Evolve a set of models with MOEAs

The content of this module can be found at `<project-dir>/src/MOEA`.

This module aims at getting the results of the bug report processing and inferring models using MOEAS.
To run this functionality, use the script `PymooSingleRun.py` in the root directory of this project.
This script receives a few parameters (in this order), as described below:

1) program name (see 'data/bug/dataset' for available programs in '.pkl' format);
2) independent run ID;
3) algorithm ('NSGAII', 'NSGAIII', or 'MOEAD');
4) objectives ('UnRec-Size', 'UnObs-Size', 'UnRec-UnObs', or 'UnRec-Size-UnObs'). Bear in mind that 2 objectives are only implemented with NSGAII. Default: 'UnRec-Size-UnObs';
5) random seed

For instance, if you would like to run the 20th independent run of NSGA-II with "UnRec-Size-UnObs" as objectives, using "krita" as a program and 90 as random seed, use the following command:

```bash
python PymooSingleRun.py krita 20 NSGAII UnRec-Size-UnObs 90
```

The resulting models of this execution will be outputted to `results/krita/NSGAII/UnRec-Size-UnObs/*.pkl`, alongside some logs and time results.