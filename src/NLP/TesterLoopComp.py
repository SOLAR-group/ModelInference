from src.NLP.BugzillaCall import *
import time
import re
import xlsxwriter

'''
    Use this script to just create an excel file with the statistics for every component of the defined Product, 
    or to create also a dataset for every one of them (set the latest parameter of the bugzilla functions).
    
    You have to set the variable Product and the relative bugzilla website
'''

#Most famous project that use bugzilla:
#     Mozilla:        https://bugzilla.mozilla.org/             #The search of the program sometime return server error
#     GNOME:          https://bugzilla.gnome.org/
#     KDE:            https://bugs.kde.org/
#     Apache Project: https://bz.apache.org/bugzilla/
#     LibreOffice:    https://bugs.documentfoundation.org/    Product=LibreOffice&&Comp=Writer,Draw,Calc,Impress,Charts
#     Eclipse:        https://bugs.eclipse.org/bugs/
#    Linux Distributions
#     Gentoo:         https://bugs.gentoo.org/
#     Novell:         https://bugzilla.novell.com/


#Set the specific bugzilla website and (optional) authentication credentials
setting = {
    "url_jsonrpc": "https://bugzilla.mozilla.org/jsonrpc.cgi",
    "bugzilla_auth": {
        "login": "i",
        "password": "U"
}}

Bugzilla = BugzillaCall(setting)

Bugzilla.checkDir()

flagComponent = 1   #search all component for a specific program, compulsory for the search based on all component
flagCompute   = 1     #Compute all the pipeline
flagJustParsing = 0

listProd = []
row = 0
col = 0

name = re.search('[\.][a-zA-Z0-9_]*[\.]', setting["url_jsonrpc"]).group(0).replace(".", "")
if name is None:
    name = re.search('[////][a-zA-Z0-9_]*[\.]', setting["url_jsonrpc"]).group(0).replace(".", "").replace("/", "")

startfull = time.time()

# Variable to set
product = "Thunderbird"

print("Product: "+product)

fail = open('data/bug/Fail_' + product+ '.txt', 'w', 1)
workbook = xlsxwriter.Workbook('data/Loop/BestC_'+name+'_'+product+'.xlsx')
worksheet = workbook.add_worksheet()

if flagComponent:
    listComp = Bugzilla.findProductComponent(product, 0)
    print("Num. components: "+str(len(listComp))+"\n")

listComp.sort()

for cmp in listComp:
    component = cmp
    print("Component: "+ component)
    n_jobs = 6    # Number of Core/Job
    limit = 0       # Max number of Bugs to receive in response
    resolution = ['FIXED', 'WONTFIX', 'WORKSFORME', 'DUPLICATE'] # Search Bug parameters
    date_start = "2000-01-01"

    start = time.time()
    result = []
    try:
        if flagCompute:
            # Search all the bug, based on the previous parameters
            listBugID = Bugzilla.findBugIDParallel(product, date_start, limit, component, resolution, 1, n_jobs)

            # Get all comment of the previous resulted bug
            commentBug = Bugzilla.findBugCommentParallel(listBugID, product, component, 0, 1, n_jobs)

            # Search on the first comment of every bug the steps to reproduce it
            result = Bugzilla.findStepsSeqPar(product, component, listBugID, commentBug, 1, 1)

        if flagJustParsing:
            # Setting flagCompute=0 avoid to research bug and comment, but use previously the stored file
            result = Bugzilla.findStepsSeqPar(product, component, -1, None, 0, 0)
    except KeyboardInterrupt:
        workbook.close()
    except:
        print("Server error")
        fail.write(product + "\n")
        continue

    end = time.time()
    print('\nTime tot. {}:{} \n'.format(*divmod(round(end - start), 60)))

    worksheet.write(row, col, len(result))
    worksheet.write(row, col + 1, component)
    if flagCompute:
        worksheet.write(row, col + 2, len(listBugID))
    row += 1
    print("---------")

workbook.close()
endfull = time.time()

print('\nTime loop tot. {}:{}'.format(*divmod(round(endfull - startfull), 60)))