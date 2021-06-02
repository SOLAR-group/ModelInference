from src.NLP.BugzillaCall import *
from src.NLP.TextWork import *
import numpy as np
import gc
import nltk

nltk.download('punkt')
nltk.download('stopwords')

'''
    Use this script 'mainly' to retrieve the bugs of a predefined Product and component, and then compute the clustering
    phase.
    If you already have the dataset file, set all the flag to 0 except flagTextCompClst, and set the variable Product 
    and component accordingly.
'''

# Most famous project that use bugzilla:
#     Mozilla:        https://bugzilla.mozilla.org/             #The search of the program sometime return server error
#     GNOME:          https://bugzilla.gnome.org/
#     KDE:            https://bugs.kde.org/
#     Apache Project: https://bz.apache.org/bugzilla/
#     LibreOffice:    https://bugs.documentfoundation.org/  #Product=LibreOffice := Comp=Writer,Draw,Calc,Impress,Charts
#     Eclipse:        https://bugs.eclipse.org/bugs/
# Linux Distributions
#     Gentoo:         https://bugs.gentoo.org/
#     Novell:         https://bugzilla.novell.com/

# Set the specific bugzilla website and (optional) authentication credentials
setting = {
    "url_jsonrpc": "https://bugzilla.mozilla.org/jsonrpc.cgi",
    "bugzilla_auth": {
        "login": "i",
        "password": "U"}
}

Bugzilla = BugzillaCall(setting)

Bugzilla.checkDir()

flagProduct = 0
flagComponent = 0   # Search all component for a specific program, compulsory for the search based on all component
flagComputeSearch = 0     # Retrieve the bugs
flagJustParsing = 0       # Used to only test different parsing format of the comment text
flagTextCompClst = 1      # Compute the clustering phase

start = time.time()
listComp = []

# Return a list of the available Product on the bugtracker website,
# if already know the program you can set the flagPr to 0
if flagProduct:
    listProd = Bugzilla.findProd()
else:
    product = "krita"

print("Product: "+product)

# Return a list of the components of the specified program,
# set flagComponent to 1 to make the next search based on all component
if flagComponent:
    listComp = Bugzilla.findProductComponent(product, 1)
    print("Num. components: "+str(len(listComp))+"\n")

n_jobs = 6    # Number of Core/Job
limit = 0       # Max number of Bugs to receive in response, 0 to search all
resolution = ['FIXED', 'WONTFIX', 'WORKSFORME', 'DUPLICATE']
date_start = "2000-01-01"

if not listComp:
    component = "plus"  # used in TextCompClst to find the already stored file
else:
    component = listComp  # used in ComputeSearch

# If you want to search based on a specific component, decomment next line of code
component = "plus"

result = {}

if flagComputeSearch:
    # Search all the bug, based on the previous parameters
    # listBugID = Bugzilla.findBugID(program, date_start, limit, component, resolution, 0, 0)
    listBugID = Bugzilla.findBugIDParallel(product, date_start, limit, component, resolution, 1, n_jobs)

    # Get all comment of the previous resulted bug
    commentBug = Bugzilla.findBugCommentParallel(listBugID, product, component, 0, 1, n_jobs)

    # Search on the first comment of every bug the steps to reproduce it
    result = Bugzilla.findStepsSeqPar(product, component, listBugID, commentBug,  1, 1)

if flagJustParsing:
    # Setting flagComputeSearch=0 avoid to research bug and comment, but use the previously stored file
    result = Bugzilla.findStepsSeqPar(product, component, -1, None, 1, 0)

if isinstance(component, list):
    component = "plus"

gc.collect()

if flagTextCompClst:
    twork = None

    # For dataset with very higher number of bug this value need to be increased (0.72), in order to avoid the creation
    # of one cluster much bigger than others
    threshold = np.float32(0.70)

    if flagJustParsing or flagComputeSearch:
        twork = TextWork(product, component, threshold, result)
    else:
        twork = TextWork(product, component, threshold, None)  # Use stored file of Prod and Comp

    twork.pre_process()

    twork.Corpus_Cluster()

    twork.mapping1stPhase()

    twork.mapping2stPhase()

end = time.time()
print('\nTime tot. {}:{} \n'.format(*divmod(round(end - start), 60)))