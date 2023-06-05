#!/bin/bash
##############################################################################
# ESGF credentials login script
#
###############################################################################

# Deactivate any environment currently in use
conda deactivate

# Activate the environment needed to log-in
conda activate esgf-pyclient

# Run the python login script
login_dir='/glade/u/home/jonahshaw/Scripts/git_repos/internalvar-vs-obsunc/esgf_login'
python $login_dir/esgf_login.py