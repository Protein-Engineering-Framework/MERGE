This repository contains supplementary information to

Alexander-Maurice Illig<sup>*1,§*</sup>, Niklas E. Siedhoff<sup>*1,§*</sup>, Ulrich Schwaneberg<sup>*1,2*</sup>, Mehdi D. Davari<sup>*3,\**</sup>, <br>
A hybrid model combining evolutionary probability and machine learning leverages data-driven protein engineering, *To be published*<br>
Preprint available at bioRxiv: https://www.biorxiv.org/content/10.1101/2022.06.07.495081v1

<sup>*1*</sup><sub>Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany</sub> <br>
<sup>*2*</sup><sub>DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany</sub> <br>
<sup>*3*</sup><sub>Department of Bioorganic Chemistry, Leibniz Institute of Plant Biochemistry, Weinberg 3, 06120 Halle, Germany</sub> <br>
<sup>*\**</sup><sub>Corresponding author</sub> <br>
<sup>*§*</sup><sub>Equal contribution</sub> <br>


# Hybrid_Model
An evolutionary probability-based hybrid model to support protein engineering campaigns by predicting the fitness of variants based on their sequence.

This repository contains the [source files](/Examples/scripts) to reproduce the results of our manuscript using the form of protein sequence encoding in combination with the hybrid (statistical energy of a DCA model/predicted fitness of a trained supervised regression model) prediction presented using two sequence-fitness datasets as examples of a ["low-*N*"](/Examples/example_rl401.ipynb) and a ["substitutional extrapolation"](/Examples/example_pabp.ipynb) protein engineering task.
To reproduce the results of the example, run the provided Jupyter Notebooks. The ["substitutional extrapolation"](/Examples/example_pabp.ipynb) notebook also contains commands for preprocessing tasks required to create a hybrid model.

Using our protein engineering framework [PyPEF](https://github.com/Protein-Engineering-Framework/PyPEF), a simplified application of hybrid modeling alongside other machine learning-based modeling methods is possible.
