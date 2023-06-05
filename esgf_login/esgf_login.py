from pyesgf.logon import LogonManager

def login_esgf():
    
    lm = LogonManager()
    lm.logoff()
#    print(lm.is_logged_on())
    
    OPENID = 'https://esgf-node.llnl.gov/esgf-idp/openid/jshaw35'
    lm.logon_with_openid(openid=OPENID, password=None, bootstrap=True)
    
#    print(lm.is_logged_on())
    
    # This is logging me on every time regardless of whether I put in the correct password...
    if (lm.is_logged_on() == True): print('ESGF login successful.')
    
if __name__ == "__main__":
    
    login_esgf()
    