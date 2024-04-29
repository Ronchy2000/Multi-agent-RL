# Team-Coordination on Graphs with Risky Edges (TCGRE)
This is implementation of team-coordination between multiple agents in presence of adversaries using graphs as mentioned in our paper:

Manshi Limbu, Sara Oughourli, Zechen Hu, Xuan Wang, Xuesu Xiao, Daigo Shishika, [Team Coordination on Graphs with State-Dependent Edge Costs](https://ieeexplore.ieee.org/abstract/document/10341820)

[Youtube Link is here!](https://www.youtube.com/watch?v=UnMjOX3ffw8&ab_channel=DaigoShishika)

## Requirements
* networkx(>=3.0)
* numpy(>=1.24.2)
* matplotlib(>=3.7.0)


# Installation

```bash
pip3 install -r requirements.txt
```


## Run the demo (small nodes graph)

```bash
cd team-coordination
python3 quickCompare.py
```

## Run the demo (large nodes graph)
For `--nodes` param, add number of nodes and for `riskyedge` param, add risk edge ratio (more info in the paper). 

```bash
cd team-coordination
python3 graphCompare.py --nodes 10 --riskedge 0.2
```

## Algorithms 
You can choose between following algorithms:
* `jsg`: Converts multi-agent problem as single agent path planning algorithm. 

* `cjsg`: Heirarchial path planning algorithm that alleviates the curse of dimesionality casued by `jsg`. 


## Cite

Please cite our paper if you use this code in your own work:
```
@INPROCEEDINGS{10341820,
  author={Limbu, Manshi and Hu, Zechen and Oughourli, Sara and Wang, Xuan and Xiao, Xuesu and Shishika, Daigo},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Team Coordination on Graphs with State-Dependent Edge Costs}, 
  year={2023},
  volume={},
  number={},
  pages={679-684},
  keywords={Costs;Statistical analysis;Scalability;Path planning;Planning;Complexity theory;Game theory},
  doi={10.1109/IROS55552.2023.10341820}}

```
