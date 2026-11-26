## Muography Forward Modelling Toolkit
## geol-vector-muon-opacities

This is a small supplementary toolkit to accompany the paper "Muographic corrections and methods for geological interpretation with a single muon detector source".
The purpose is to be enable recreating the model cases for figures 5-8, and encourage exploration of the geological cases and the proposed methods of filtering and analysing the model results.

## Description

stratmodel.py is for creating simple surface-based stratigraphic geological models, and strat_raytrace.py is for calculating density integrals along ray paths through these model surfaces. healpix_detector.py is for creating a detector object to generate and store ray path vectors, and to plot results with plot_sky(). strat_plot.py contains a function for plotting sections through the stratmodel layer models.

Steps to recreate the figures using these scripts are in the strat_modelling_fig5-8.ipynb Jupyter notebook.
An interactive tool called TiltExplorer is included in the Jupyter notebook for investigating varying the angle applied in the Equation 2.5 perspective correction.

## Authors

Tom J. McNamara
(assisted by Anthropic's Claude LLM)
