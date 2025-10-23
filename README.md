# Computational-Detection-of-Syllable-Boundaries
Code for the paper "Computational Detection of Syllable Boundaries in Undeciphered Languages Using Segment Surprisal"
The complementary data files are hosted at: 10.5281/zenodo.17418251
The online appendix is included above.

**Stan model**
1. Run Stanmodel_dataprep.R to generate the Stan data from the data file
2. Run the model with Model_run.R which reads in syll_newmodel_alterparam.stan
3. Process and display the results with Bayesian_postprocessing.R

**Neural Net model**
The neural net models and experiments are currently saved in the original jupyter notebooks used in the development of the paper. These are being edited and reorganized to facilitate easier understanding and reproduction. If you have questions in the meantime, please contact the paper's authors.