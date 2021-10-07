# Matrix Factorisation Leveraging Similarity Matrices

Contains code for the similarity matrices for both drugs and targets

## Similarity Matrices Calculation
- Drug-Drug similarity is found via the Dice similarity score between the Morgan fingerprints of the drugs
- Target-Target similarity is found using a normalised smith-waterman score

## Inspiration
The original paper that we draw inspiration from can be found [here](https://www.bic.kyoto-u.ac.jp/pathway/Files/kdd13.pdf). A notable difference is that there is only one similarity matrix for drugs and one for targets as opposed to the five similarity matrices for the original paper.
