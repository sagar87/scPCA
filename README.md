
# scPCA - A probabilistic factor model for single-cell data

![pypi](https://img.shields.io/pypi/v/scpca.svg)
![release workflow](https://github.com/sagar87/scPCA/actions/workflows/release.yaml/badge.svg)
![push workflow](https://github.com/sagar87/scPCA/actions/workflows/branch.yaml/badge.svg)

scPCA is a versatile matrix factorisation framework designed to analyze single-cell data across diverse experimental designs.

![scPCA schematic](https://github.com/sagar87/scPCA/blob/main/docs/scpca_schematic.png?raw=true)

*scPCA is a young project and breaking changes are to be expected.*

## scPCA in a nutshell

scPCA enables the analysis of single-cell RNA-seq data across condtions. In simple words, it enables the incorporation of a design (model) matrix that encodes the experimental design of the dataset and infers how the gene loading weight vectors change from a specified reference condition to the treated condtion. 

https://github.com/user-attachments/assets/182af56e-14e0-4357-ab31-1b392dd45d18

## Quick install

scPCA makes use `torch`, `pyro` and `anndata`. We highly recommend to run scPCA on a GPU device.

### Via Pypi

The easiest option to install `scpca` is via Pypi. Simply type

```
$ pip install scpca
```


into your shell and hit enter.

* Free software: MIT license
* Documentation: https://sagar87.github.io/scPCA/index.html

## Credits

* Harald Vöhringer
