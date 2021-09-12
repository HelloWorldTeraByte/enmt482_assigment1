#        >(')____,  >(')____,  >(')____,  >(')____,  >(') ___,
#         (` =~~/    (` =~~/    (` =~~/    (` =~~/    (` =~~/
#    ~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~^`---'~^~^~
#************************************************************************/
#*                                                                      */
#                            sensor_model_compare.py 
#*                                                                      */
#************************************************************************/


#   Authors:        Jason Ui        
#                   Randipa
#
#
#
#   Date created:       25/08/2021
#   Date Last Modified: 25/08/2021


#************************************************************************/

#  Module Description:
#  
#

#Enter in Terminal "pip3 install tabulate"
from tabulate import tabulate
import src.robust_lsr_linear as rlsr_lin
import src.robust_lsr_nonlinear as rlsr_nlin
import src.robust_lsr_polynomial as rlsr_pol
import src.robust_lsr_piecewise as rlsr_pw

table = [
        ["Sonar1","",""],
        ["Method","Mean","Variance"],
        ["RLSR (Linear Model)",rlsr_lin.err_mse,rlsr_lin.err_var], 
        ["RANSAC (Linear Model)",'X','X'],
        ["----------","----------","----------"],
        ["IR3","",""],
        ["Method","Mean","Variance"],
        ["RLSR (Non-Linear Model)",rlsr_nlin.err_mse,rlsr_nlin.err_var], 
        ["RLSR (Polynomial Model)",rlsr_pol.err_mse,rlsr_pol.err_var], 
        ["RLSR (Piecewise)",rlsr_pw.err_mean,rlsr_pw.err_var], 
        ["RANSAC (Polynomial Model)",'X','X'],
        ["RANSAC (Piecewise)",'X','X'],
        ]

print(tabulate(table))