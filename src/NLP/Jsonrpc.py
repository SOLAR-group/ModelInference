import json
import sys
import traceback
import requests


def try_rpc(g_query, maxtry, *args, **kw):
    for i in range(1, maxtry + 1):
        try:
            return g_query(*args, **kw)  #just call the func get_query
        except:
            print("".join(traceback.format_exception(*sys.exc_info())))
    exit("No more coins")



def get_query(setting, method, **params):

    #params["Bugzilla_login"] = setting["bugzilla_auth"]["login"]
    #params["Bugzilla_password"] = setting["bugzilla_auth"]["password"]

    serveResponse = requests.get(
        params={"method": method,
                "params": json.dumps([params])
                },
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
        url=setting["url_jsonrpc"])

    serveResponse.raise_for_status()     # server error / bad request / etc.
    readall = serveResponse.json()

    if readall.get("error"):       # regular error message
        raise Exception(readall)

    return readall
