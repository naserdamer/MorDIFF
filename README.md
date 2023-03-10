# MorDIFF
Official repository of the paper **MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

[*** Accepted at the 11th International Workshop on Biometrics and Forensics 2023 (IWBF 2023) ***](https://warwick.ac.uk/fac/sci/dcs/people/victor_sanchez/iwbf2023/)

![grafik](https://user-images.githubusercontent.com/85616215/217528980-9bbe613e-baa9-4624-8163-9666c972c1c2.png)




Paper available under this [LINK](https://arxiv.org/abs/2302.01843)

The benchmark SYN-MAD 2022 data is available under: [REPO+DATA](https://github.com/marcohuber/SYN-MAD-2022) + [PAPER](https://ieeexplore.ieee.org/document/10007950)

Morphing script is provided.

The MorDIFF data can be downloaded from this [LINK](https://drive.google.com/file/d/1t8Jw1FoBDOXvMxDJ4uJZeffsCXcV8mMe/view?usp=share_link) (please share your name, affiliation, and official email in the request form).


**Citation:**

If you use MorDIFF datas, please cite the following [paper](https://arxiv.org/abs/2302.01843):

```
@misc{MorDIFF,
  doi = {10.48550/ARXIV.2302.01843},
  url = {https://arxiv.org/abs/2302.01843},
  author = {Damer, Naser and Fang, Meiling and Siebke, Patrick and Kolf, 
  Jan Niklas and Huber, Marco and Boutros, Fadi},
  title = {MorDIFF: Recognition Vulnerability and Attack Detectability of Face 
  Morphing Attacks Created by Diffusion Autoencoders},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

and the base benchmark, the SYN-MAD 2022 [paper](https://ieeexplore.ieee.org/document/10007950):

```
@inproceedings{SYNMAD2022,
  author    = {Marco Huber and
               Fadi Boutros and
               Anh Thi Luu and
               Kiran B. Raja and
               Raghavendra Ramachandra and
               Naser Damer and
               Pedro C. Neto and
               Tiago Gon{\c{c}}alves and
               Ana F. Sequeira and
               Jaime S. Cardoso and
               Jo{\~{a}}o Tremo{\c{c}}o and
               Miguel Louren{\c{c}}o and
               Sergio Serra and
               Eduardo Cerme{\~{n}}o and
               Marija Ivanovska and
               Borut Batagelj and
               Andrej Kronovsek and
               Peter Peer and
               Vitomir Struc},
  title     = {{SYN-MAD} 2022: Competition on Face Morphing Attack Detection Based
               on Privacy-aware Synthetic Training Data},
  booktitle = {{IEEE} International Joint Conference on Biometrics, {IJCB} 2022,
               Abu Dhabi, United Arab Emirates, October 10-13, 2022},
  pages     = {1--10},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/IJCB54206.2022.10007950},
  doi       = {10.1109/IJCB54206.2022.10007950},
}
```

To use the provided morphing script, you need to download the pre-trained model from Preechakul et al. [REPO](https://github.com/phizaz/diffae). If you use the provided morphing script with the pre-trained model provided by Preechakul et al. ([paper](https://arxiv.org/abs/2111.15640)) please additionally follow their citation requierments ([REPO](https://github.com/phizaz/diffae)).

The MorDIFF data, such as the SYN-MAD 2022 benchmark, is based on the Face Research Lab London dataset (FRLL). Please follow all the requierments of the [FRLL dataset](https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666).


##

**License:**

The dataset, the implementation, or trained models, use is restricted to research purpuses. The use of the dataset or the implementation/trained models for product development or product competetions (incl. NIST FRVT MORPH) is not allowed.
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.

