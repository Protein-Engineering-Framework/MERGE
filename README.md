This repository contains supplementary information to

Alexander-Maurice Illig<sup>*1,§*</sup>, Niklas E. Siedhoff<sup>*1,§*</sup>, Ulrich Schwaneberg<sup>*1,2*</sup>, Mehdi D. Davari<sup>*3,\**</sup>, <br>
A hybrid model combining evolutionary probability and machine learning leverages data-driven protein engineering *(Working title; To be published)*<br>
Preprint available at bioRxiv: https://www.biorxiv.org/content/10.1101/2022.06.07.495081v1

<sup>*1*</sup><sub>Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany</sub> <br>
<sup>*2*</sup><sub>DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany</sub> <br>
<sup>*3*</sup><sub>Department of Bioorganic Chemistry, Leibniz Institute of Plant Biochemistry, Weinberg 3, 06120 Halle, Germany</sub> <br>
<sup>*\**</sup><sub>Corresponding author</sub> <br>
<sup>*§*</sup><sub>Equal contribution</sub> <br>


# MERGE  
A hybrid method (<ins>MERGE</ins>) combining evolutionary probability and <ins>m</ins>achine l<ins>e</ins>arning leve<ins>r</ins>a<ins>ge</ins>s data-driven protein engineering by enabling trustworthy predicting the fitness of variants based on their sequence, even in low-*N* data situations.

This repository contains the [source files](/Examples/scripts) to reproduce the results of our manuscript using the form of protein sequence encoding in combination with the hybrid (statistical energy of a DCA model/predicted fitness of a trained supervised regression model) prediction presented using two sequence-fitness datasets as examples of a ["low-*N*"](/Examples/example_rl401.ipynb) and a ["substitutional extrapolation"](/Examples/example_pabp.ipynb) protein engineering task.
To reproduce the results of the example, run the provided Jupyter notebooks. The ["substitutional extrapolation"](/Examples/example_pabp.ipynb) notebook also contains commands for preprocessing tasks required to create a hybrid model.
For all datasets studied, already encoded datasets containing the variant identifiers, the corresponding fitness values, and the encoded sequences are provided as CSV files and used wild-type sequences are provided as FASTA files (see [Data](https://github.com/Protein-Engineering-Framework/MERGE/tree/main/Data)).

## Framework Implementation
Using our protein engineering framework [PyPEF](https://github.com/Protein-Engineering-Framework/PyPEF), a simplified application of the MERGE hybrid method alongside other encoding and machine learning-based modeling methods is possible. Variant-fitness datasets that can be used for encoding and modeling with PyPEF are provided at [Data/_variant_fitness_wtseq](https://github.com/Protein-Engineering-Framework/MERGE/tree/main/Data/_variant_fitness_wtseq).
