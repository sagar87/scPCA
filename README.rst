#########################################################
scPCA - A probabilistic factor model for single-cell data
#########################################################


.. image:: https://img.shields.io/pypi/v/scpca.svg
        :target: https://pypi.python.org/pypi/scpca

![scPCA schematic](https://github.com/sagar87/scPCA/blob/main/docs/scpca_schematic.png?raw=true)


scPCA is a versatile matrix factorisation framework designed to analyze single-cell data across diverse experimental designs.

*scPCA is a young project and breaking changes are to be expected.*

*************
Quick install
*************

scPCA makes use :code:`torch`, :code:`pyro` and :code:`anndata`. We highly recommend to run scPCA on a GPU device.

Via Pypi
========

The easiest option to install :code:`scpca` is throug Pypi. Simply type

.. code-block:: console

    $ pip install scpca

into your shell and hit enter.

* Free software: MIT license
* Documentation: https://sagar87.github.io/scPCA/index.html

***************
Getting started
***************


Step 1: Data preparation
========================

Running scPCA involves three steps

Preparing input data using docker
---------------------------------

We provide a docker image that contains all :code:`R` and :code:`bioconductor` dependencies to create the variant tensor and the other mutation type matrix. To use it, pull the image from docker. Note that the image is approximately 5 GB large.

.. code-block:: console

    $ docker pull sagar87/tensorsignatures-data:latest

To use the image switch into the folder containing your VCF data. Then run image using the following command and supply the VCF files as well as the name of the :code:`hdf5` output file (must be the last argument) as arguments.

.. code-block:: console

    $ docker run -v $PWD:/usr/src/app/mount sagar87/tensorsignatures-data <vcf1.vcf> <vcf2.vcf> ... <vcfn.vcf> <output.h5>

Then continue with Step 2.

Preparing input data using a custom installation
------------------------------------------------

Make sure you have :code:`R3.4.x` (!) and the packages :code:`VariantAnnotation` and :code:`rhdf5` installed. You can install them, if necessary, by executing


.. code-block:: console

    $ Rscript -e "source('https://bioconductor.org/biocLite.R'); biocLite('VariantAnnotation')"

and

.. code-block:: console

    $ Rscript -e "source('https://bioconductor.org/biocLite.R'); biocLite('rhdf5')"

from your command line.

To get started, download the following files and place them in the same directory:

`Constants.RData <http://193.62.55.163/file/R/constants.RData>`_ (contains :code:`GRanges` objects that annotate transcription/replication orientation, nucleosomal and epigenetic states)

`mutations.R <http://193.62.55.163/file/R/mutations.R>`_ (all required functions to partiton SNVs, MNVs and indels)

`processVcf.R <http://193.62.55.163/file/R/processVcf.R>`_ (loads :code:`vcf` files and creates the SNV count tensor, MNV and indel count matrix; eventually needs custom modification to make the script run on your vcfs.)

`genome.zip <http://193.62.55.163/file/R/genome.zip>`_ .


To obtain the SNV count tensor and the matrices containing other mutation types, execute :code:`processVcf.R` and pass the VCF files you want to convert, as well as a name for an output :code:`hdf5` file as command line arguments, e.g.

.. code-block:: console

    $ Rscript processVcf.R <vcf1.vcf> <vcf2.vcf> ... <vcfn.vcf> <output.h5>

In case of errors please check wether you have correctly specified paths in line 6-8. Also, take a look at the :code:`readVcfSave` function and adjust it when it fails.

Step 2: Computing trinucleotide normalisation
=============================================

TensorSignatures requires a trinucleotide normalisation constant to account for differences in the nucleotide composition of genomic states. To compute it, invoke the prep sub routine of TensorSignatures and pass the :code:`hd5` file from Step 1 as well as the path for the output file as positional arguments to the programme.

.. code-block:: console

    $ tensorsignatures prep <output.h5> <tsdata.h5>


Step 3: Run TensorSignatures
============================

There are two ways to run TensorSignatures using either the :code:`refit` option, which fits the exposures of a set of pre-defined signatures extracted from the PCAWG cohort to a your dataset, or via the :code:`train` subroutine, that performs a denovo extraction of tensor signatures. Refitting tensor signatures is computationally fast but does not allow to discover new signatures, while extracting new signatures from scratch is computationally intensive (GPU required) and requires ideally larger numbers of samples. For most use cases, with a small number of samples, we advice to use the refit option:

.. code-block:: console

    $ tensorsignatures --verbose refit tsData.h5 refit.pkl -n

To run a denovo extraction use

.. code-block:: console

    $ tensorsignatures --verbose train tsData.h5 denovo.pkl <rank> -k <size> -n -ep <epochs>


where :code:`rank` specifies the decomposition rank, :code:`size` controls the dispersion of the model, and :code:`epochs` the number of desired epochs to fit the model. TensorSignatures outputs value of the objective function (log likelihood) that is minimised during training as well as the change of the objective during an epoch interval (:code:`delta`). When deciding on the number of epochs to train the model ensure that it is sufficiently large such that the objective function converges, i.e. the :code:`delta` value is close to, or fluctuates around zero. For more information on how to run TensorSignatures in a practical setting see the documentation. Running TensorSignatures will yield a pickle dump which can subsequently inspected using the tensorsignatures package.


********
Features
********

* Run :code:`tensorsignatures` on your dataset using the :code:`TensorSignature` class provided by the package or via the command line tool.
* Compute percentile based bootstrap confidence intervals for inferred parameters.
* Basic plotting tools to visualize tensor signatures and inferred parameters

*******
Credits
*******

* Harald Vöhringer and Moritz Gerstung
