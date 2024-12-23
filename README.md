# A Review of Excited State Eigensolvers

This repository accompanies the paper "A Review of Excited State Variational Eigensolvers" posted to ArXiv.  This repository contains both example scripts for simple implimentations of the excited state algorithms covered in the review, and scripts for benchmarking these algorithms on molecules such as $H_2$ or $LiH$.

Excited state eigensolvers consist of two building blocks: a ground state eigensolver, and an excited state modification.  The ground state eigensolver is able to find the lowest eigenvalue accessible by the provided ansatz, and the excited state modification changes the loss function to force the ground state eigensolver to discover excited states.  As an example, the Variational Quantum Deflation (VQD) algorithm uses the variational quantum eigensolver to discover the lowest energy eigenstate, and then modfies the loss function to effectively raise the energy of this lowest eigenvalue.  When the ground state eigensolver is applied a second time, it will now find the first excited state rather than the ground state.

## Ground State Eigensolvers
Many ground state eigensolvers have been discovered.  The most popular is the Variational Quantum Eigensolver (Peruzzo et. al. 2014), which has been applied to many molecules and quantum systems.  

The ground state algorithms covered in this review are:
1. Variational Quantum Eigensolver (VQE)
2. Variational Quantum Imaginary Time Evolution (VarQITE)
3. Trotterized Quantum Imaginary Time Evolution (TrotterQITE)
4. Contracted Quantum Eigensolver (CQE)
5. Fully Quantum Eigensolver (FQE)
6. Quantum Phase Estimation (QPE)

## Excited State Modifications
Several Excited State Modifications have been discovered.  The most well-known is Deflation, where a term is added to the loss function of the variational algorithm, which increases the energy of the discovered ground state, allowing for a ground state eigensolver to find higher energy states.

The Excited-State modifications covered in this review are:
1. Deflation
2. Subspace Search
3. Subspace Expansion
4. Folded Spectrum
5. Variance

## Excited State Algorithms
Through combining a ground state algorithm and an excited state modification, an excited state algorithm can be created.  Many more excited state algorithms are possible by combining one given ground state algorithm with an excited state modification.  Only the most popular excited state methods at the time of writing are given here as examples.

The excited state algorithms contained in this review are:

1. Variational Quantum Deflation (VQD)
2. Subspace-Search Variational Quantum Eigensolver (SSVQE)
3. Quantum Imaginary Time Evolution Deflation (QITED)
4. Subspace Search Quantum Imaginary Time Evolution (SSQITE)
5. Quantum Lanczos Algorithm (QLanczos)
6. Quantum Davidson Algorithm (QDavidson)
7. Excited State Contracted Quantum Eigensolver (ES-CQE) - uses deflation
8. Fully Quantum Excited State Eigensolver (FQESS) - uses deflation
