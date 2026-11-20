.. _kw_mc:
.. index::
   single: mc (keyword in run.in)

:attr:`mc`
==========

The :attr:`mc` keyword is used to carry out Monte Carlo (:term:`MC`) trial steps, usually in combination with a :term:`MD` simulation.
Four :term:`MC` ensembles are supported, including the canonical, the semi-grand canonical (:term:`SGC`), the variance-constrained semi-grand canonical (:term:`VCSGC`) [Sadigh2012a]_ [Sadigh2012b]_, and the grand canonical ensemble.

This keyword requires an ordinary energy NEP (``model nep``) in the
:ref:`nep.txt format <nep_txt>`.

Syntax
------

::

    mc canonical <md_steps> <mc_trials> <T_i> <T_f> [group <grouping_method> <group_id>]
    mc sgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <mu_0> <species_1> <mu_1> ...} [group <grouping_method>  <group_id>]
    mc vcsgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <phi_0> <species_1> <phi_1> ...} <kappa> [group <grouping_method>  <group_id>]
    mc gcmc <md_steps> <T_i> <T_f> <num_sweeps> <a_plus> <n_sep> <max_ghost> <num_species> {<species_0> <mu_0> <mass_0> <e0_0> ...} [parallel] [seed <seed>] [overlap_cutoff <num> {<species> <d>} ...]

Parameters
----------

For the :attr:`canonical`, :attr:`sgc`, and :attr:`vcsgc` forms, ``mc_trials`` trials are performed every ``md_steps`` MD steps.
The parameters of the :attr:`gcmc` form are described in its own section below.
The MC target temperature changes linearly from ``T_i`` to ``T_f`` during the run.

* The listed species must be supported by the :term:`NEP` model.

* For the :attr:`canonical`, :attr:`sgc`, and :attr:`vcsgc` ensembles, there is an option to specify the grouping method :attr:`grouping_method` and the group ID :attr:`group_id` in the given grouping method, after the parameter :attr:`group`.
  The functionality is illustrated in the example section below.

* For the :attr:`canonical`, :attr:`sgc`, and :attr:`vcsgc` ensembles, there must be at least one listed species in the initial model system or specified group. For example, if you list Au and Cu for doing :term:`SGC` :term:`MC`, the system or the specified group must have some Au or Cu atoms (or both); otherwise the :term:`MC` trial cannot get started.

:attr:`canonical`
^^^^^^^^^^^^^^^^^
Canonical sampling exchanges atomic species while keeping their counts fixed.

:attr:`sgc`
^^^^^^^^^^^
SGC sampling uses species transmutations with chemical-potential differences.

:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and chemical potentials (:attr:`mu_i`) in units of eV.
The species can be listed in arbitrary order.
Note that only the differences between the chemical potentials matter.

For a trial transmutation from species :math:`a` to species :math:`b`, GPUMD uses the acceptance probability

.. math::

   P_{a\to b}=\min\left[1,\exp\left(-\frac{\Delta U+\mu_b-\mu_a}{k_{\rm B}T}\right)\right],

where :math:`\Delta U` is the change in potential energy.
Thus, increasing the input :math:`\mu_b` makes conversion into species :math:`b` less favorable.
For a convention based on :math:`U-\sum_s\mu_s^{\rm phys}N_s`, the corresponding input values are :math:`\mu_s=-\mu_s^{\rm phys}` up to a common additive constant.

:attr:`vcsgc`
^^^^^^^^^^^^^
VCSGC sampling adds a concentration bias.

:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and dimensionless concentration-bias parameters (:attr:`phi_i` = :math:`\phi_i`).
The species can be listed in arbitrary order.
Only differences between the :math:`\phi_i` parameters enter the acceptance criterion.
Next one needs to specify the (dimensionless) :attr:`kappa` parameter (:math:`\kappa`).

The :math:`\phi` and :math:`\kappa` parameters constrain the average and variance of the species concentrations, respectively.
For a binary system, a symmetric scan can use :math:`\phi_A=-\phi` and :math:`\phi_B=\phi`, varying the common scan parameter :math:`\phi` approximately from -1.2 to +1.2.
This changes :math:`\phi_B-\phi_A` from -2.4 to +2.4; it is distinct from varying only one species parameter while fixing the other.
The resulting compositions depend on the interactions and temperature and should be checked from the sampled concentrations.
Typically a :math:`\kappa` value of 100 is suitable.
If the concentration fluctuations are too large (e.g., deep with miscibility gaps) one should increase this value.

The parameter normalization differs from that used in the original papers [Sadigh2012a]_ [Sadigh2012b]_; see also [Rahm2021]_.

:attr:`gcmc`
^^^^^^^^^^^^
Grand canonical sampling inserts and deletes atoms of the listed species.

Every :attr:`md_steps` :term:`MD` steps, :attr:`num_sweeps` sweeps of insertion and deletion trials are performed, while the temperature changes linearly from :attr:`T_i` to :attr:`T_f`.
The box is divided into cells of width about the radial cutoff of the :term:`NEP` model.
In each sweep, a random set of cells at least :attr:`n_sep` cutoffs apart is chosen, and each of them makes one trial: a species is chosen at random, and an insertion at a random point of the cell is tried with probability :attr:`a_plus` (0 < :attr:`a_plus` < 1), otherwise the deletion of a random atom of that species in the cell.
The energy change is calculated with the :term:`NEP` model only.

* :attr:`max_ghost`: number of spare atom slots reserved for insertions. The number of atoms in the arrays is fixed; an empty slot (a ghost atom) has type -1 and is ignored by the :term:`NEP` model, the temperature, the :attr:`nvt_lan` thermostat, the :attr:`velocity` keyword, and the outputs of :attr:`dump_xyz`, :attr:`dump_restart`, and :attr:`dump_thermo`.
* :attr:`num_species`: number of species to be inserted and deleted.
  For each species, one specifies the chemical symbol (:attr:`species_i`), the chemical potential (:attr:`mu_i`) in eV relative to the isolated atom, the mass (:attr:`mass_i`) in Dalton used for the thermal de Broglie wavelength and for the inserted atoms, and the energy of an isolated atom given by the :term:`NEP` model (:attr:`e0_i`) in eV, or :attr:`auto` to calculate it.
* :attr:`parallel`: evaluate the trials of a sweep in parallel on the GPU.
* :attr:`seed`: seed of the random numbers.
* :attr:`overlap_cutoff`: an insertion of species :attr:`species` closer than :attr:`d` (in Å) to another atom is rejected before the energy is calculated. :attr:`d` must not exceed the radial cutoff of the :term:`NEP` model.

The :attr:`mc gcmc` line must appear before :attr:`potential` in the first run, because it adds the ghost atoms.
It has to be given again in each run; the ghost atoms are added only once.
Ghost atoms and inserted atoms belong to group 0 of every grouping method.
Among the thermostats with random forces, only :attr:`nvt_lan` is supported.
Spin models are not supported.
The keywords :attr:`correct_velocity`, :attr:`compute_rdf`, and the outputs other than those listed above do not skip ghost atoms.
A line in :ref:`run.in <run_in>` can have at most 32 tokens, which limits the number of species and options in one :attr:`mc gcmc` line.

Output
------

Monte Carlo statistics are written to :ref:`mcmd.out <mcmd_out>`.

Examples
--------

Example 1
^^^^^^^^^

An example for sampling in the canonical ensemble is::

  mc canonical 100 200 500 100 group 1 3

This means

* Perform 200 :term:`MC` trials after every 100 :term:`MD` steps.
* Change the temperature for the :term:`MC` simulation linearly from 500 to 100 K.
* Only the atoms in group 3 of grouping method 1 will be considered during :term:`MC` sampling.

Example 2
^^^^^^^^^

Here is an example for :term:`MC` sampling the :term:`SGC` ensemble::

  mc sgc 100 1000 300 300 2 Cu 0 Au 0.6

This means

* Perform 1000 :term:`MC` trials after every 100 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 300 K.
* Only the Cu and Au atoms are involved in the :term:`MC` process.
  The input parameter for Au is 0.6 eV higher than that for Cu, so the parameter contribution to the acceptance exponent favors Au-to-Cu transmutations.

Example 3
^^^^^^^^^

Here is an example for sampling in the :term:`VCSGC` ensemble::

  mc vcsgc 200 1000 500 500 2 Al -2 Ag 0 10000

This means

* Perform 1000 :term:`MC` trials after every 200 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 500 K.
* Only the Al and Ag atoms are involved in the :term:`MC` process.
  The dimensionless :math:`\phi` parameters for Al and Ag are -2 and 0, respectively.
  The dimensionless :math:`\kappa` parameter is 10000.

Example 4
^^^^^^^^^

Here is an example for inserting and deleting H atoms in the grand canonical ensemble::

  mc gcmc 100 300 300 1 0.5 3 2000 1 H -2.5 1.008 auto parallel overlap_cutoff 1 H 1.0
  potential nep.txt

This means

* Perform one sweep of insertion and deletion trials of H after every 100 :term:`MD` steps at 300 K.
* Insertions and deletions are proposed with equal probabilities, in cells at least 3 cutoffs apart.
* Up to 2000 H atoms can be inserted.
* The chemical potential of H is -2.5 eV, and the energy of an isolated H atom is calculated from the :term:`NEP` model.
* Insertions closer than 1 Å to another atom are rejected.

Notes
-----

None.
