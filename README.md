FirmCore Decomposition of Multilayer Networks
================================================

This repository contains the implementation of algorithms, and used datasets of paper "FirmCore Decomposition of Multilayer Networks" (ACM The Web Conference 2022 (WWW)). 


#### Authors: [Ali Behrouz](https://abehrouz.github.io/), [Farnoosh Hashemi](https://farnooshha.github.io//), [Laks V.S. Lakshmanan](https://www.cs.ubc.ca/~laks/)
#### [Link to the paper](https://dl.acm.org/doi/10.1145/3485447.3512205) ([Arxiv](https://arxiv.org/pdf/2208.11200.pdf))
#### [Poster]()
#### [Brief video explanation]()




### New Multilayer Network Datasets
----------------  
1. Google+ (a billion-scale temporal network)



### Key Contributions
----------------  
1. We extend the notion of k-core to multilayer networks, FirmCore, with polynomial time decomposition. 
2. We extend FirmCore to directed multilayer networks.
3. We define the problem of densest subgraph for directed multilayer networks.
4. We design the **first** polynomial-time approximation algorithm for the problem of deensest subgraph for (undirected/directed) multilayer networks.




### Abstract
----------------  
A key graph mining primitive is extracting dense structures from graphs, and this has led to interesting notions such as k-cores which subsequently have been employed as building blocks for capturing the structure of complex networks and for designing efficient approximation algorithms for challenging problems such as finding the densest subgraph. In applications such as biological, social, and transportation networks, interactions between objects span multiple aspects. Multilayer (ML) networks have been proposed for accurately modeling such applications. In this paper, we present FirmCore, a new family of dense subgraphs in ML networks, and show that it satisfies many of the nice properties of k-cores in single-layer graphs. Unlike the state of the art core decomposition of ML graphs, FirmCores have a polynomial time algorithm, making them a powerful tool for understanding the structure of massive ML networks. We also extend FirmCore for directed ML graphs. We show that FirmCores and directed FirmCores can be used to obtain efficient approximation algorithms for finding the densest subgraphs of ML graphs and their directed counterparts. Our extensive experiments over several real ML graphs show that our FirmCore decomposition algorithm is significantly more efficient than known algorithms for core decompositions of ML graphs. Furthermore, it returns solutions of matching or better quality for the densest subgraph problem over (possibly directed) ML graphs.




### Code
----------------  
This folder includes the implementation of all algorithms, and datasets.




### Usage
----------------  
Run the following command from the folder [`Code/`](Code)

```
python main_FirmCore.py [-h] [--save] [--dic] [-l L] [-b B]  d m g
```

Positional arguments: 

`d` : dataset 

`m` : method {core, densest}

`g` : type of graph {directed, undirected}

Optional arguments: 
`-h, --help` : show the help message and exit 

`--save` : save results 

`-l L` : value of lambda 

`-b B` : value of beta


### Examples
----------------
1.  
```
python  main_FirmCore.py  Homo  core  undirected  --dic
```
2.  
```
python  main_FirmCore.py  Homo  core  undirected  --dic  --save
```
3.  
```
python  main_FirmCore.py  Homo  core  undirected  -l  3
```
4.  
```
python  main_FirmCore.py  Homo  densest  undirected  -b  1.1"
```


### Reference
----------------  
```
@inproceedings{FirmCore2022,
author = {Hashemi, Farnoosh and Behrouz, Ali and Lakshmanan, Laks V.S.},
title = {FirmCore Decomposition of Multilayer Networks},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3512205},
doi = {10.1145/3485447.3512205},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {1589–1600},
numpages = {12},
keywords = {multi-layer graph, k-core, Graph mining, densest subgraph.},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
```

