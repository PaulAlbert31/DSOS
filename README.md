# Official Pytorch implementation for *Addressing out-of-distribution label noise in webly-labelled data*.
Paper link: *to come*
![DSOS](https://github.com/PaulAlbert31/DSOS/DSOS.pdf)

## Set dataset location
The path to the datasets should me set in the **mypath.py** file

## Experiments
```
bash train.sh
```
You can resume a checkpoint using 
```
--resume path/to/checkpoint.pth.tar
```

## Cite the  paper
```
@inproceedings{2022_WACV_DSOS,
  title="{Addressing out-of-distribution label noise in webly-labelled data}",
  author="Albert, Paul and Ortego, Diego and Arazo, Eric and O{'}Connor, Noel and McGuinness, Kevin",
  booktitle="{Workshop on Applications of Computer Vision (WACV)}",
  year="2022"}
```