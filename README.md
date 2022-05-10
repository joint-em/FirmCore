# FirmCore

Run the following command from the folder "Code/"

" python main_FirmCore.py [-h] [--save] [--dic] [-l L] [-b B]  d m g"

positional arguments: 

d dataset 

m method {core, densest}

g type of graph {directed, undirected}

optional arguments: 
-h, --help show the help message and exit 

--save save results 

-l L value of lambda 

-b B value of beta


### Examples

"python  main_FirmCore.py  Homo  core  undirected  --dic"

"python  main_FirmCore.py  Homo  core  undirected  --dic  --save"

"python  main_FirmCore.py  Homo  core  undirected  -l  3"

"python  main_FirmCore.py  Homo  densest  undirected  -b  1.1"

