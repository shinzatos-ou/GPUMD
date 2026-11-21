.. _mcmd_out:
.. index::
   single: mcmd.out (output file)

``mcmd.out``
============

This file records MC acceptance ratios and species concentrations.
It is generated when invoking the :ref:`mc keyword <kw_mc>`.

Results are appended to the file.

File format
-----------
The data in this file are organized as follows:

* column 1: number of :term:`MD` steps
* column 2: acceptance ratio of the :term:`MC` trials
* column 3 and more: species concentrations in the specified order

For :attr:`mc gcmc`, the columns are:

* column 1: number of :term:`MD` steps
* column 2: acceptance ratio of all the insertion and deletion trials of this call
* column 3: number of trials of this call, which is :attr:`num_sweeps` times the number of cells chosen in each sweep
* column 4 and more: for each species in the specified order, the number of atoms, the acceptance ratio of the insertions, and the acceptance ratio of the deletions;
  if :attr:`overlap_cutoff` is used, the number of insertions rejected by the overlap check is added after the deletion acceptance ratio of each species

An acceptance ratio is written as 0 when no trial of that kind was made.
