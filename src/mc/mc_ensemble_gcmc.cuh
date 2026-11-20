/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "mc_ensemble.cuh"

class MC_Ensemble_GCMC : public MC_Ensemble
{
public:
  MC_Ensemble_GCMC(
    const std::vector<std::string>& tokens,
    int num_steps_mc,
    const std::string& potential_file_name);
  virtual ~MC_Ensemble_GCMC(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

  void set_gcmc_parameters(
    const std::vector<std::string>& gcmc_species,
    const std::vector<int>& gcmc_types,
    const std::vector<double>& gcmc_mu,
    const std::vector<double>& gcmc_mass,
    const std::vector<double>& gcmc_e0,
    const std::vector<int>& gcmc_e0_auto,
    int num_gcmc_sweeps,
    double gcmc_insertion_probability,
    int domain_separation_factor,
    int max_ghost_atoms,
    bool gcmc_parallel);

  void ensure_gcmc_capacity(Atom& atom, Box& box, std::vector<Group>& groups);
  static bool capacity_reserved(void);

  void set_overlap_cutoffs(const std::vector<float>& overlap_cutoff2_by_type);
  void set_seed(unsigned seed);

private:
  GPU_Vector<int> NN_ij;
  GPU_Vector<int> NL_ij;
  GPU_Vector<unsigned int> candidate_min_d2;
  GPU_Vector<int> NN_radial_after;
  GPU_Vector<int> NN_angular_after;
  GPU_Vector<float> x12_radial_after;
  GPU_Vector<float> y12_radial_after;
  GPU_Vector<float> z12_radial_after;
  GPU_Vector<float> x12_angular_after;
  GPU_Vector<float> y12_angular_after;
  GPU_Vector<float> z12_angular_after;

  std::vector<std::string> gcmc_species;
  std::vector<int> gcmc_types;
  std::vector<double> gcmc_mu;
  std::vector<double> gcmc_mass;
  std::vector<double> gcmc_e0;
  int num_gcmc_sweeps = 0;
  double gcmc_insertion_probability = 0.5;
  int domain_separation_factor = 3;
  int max_ghost_atoms = 0;
  std::vector<float> overlap_cutoff2_by_type;

  bool format_header_written = false;
  bool gcmc_capacity_warned = false;
  bool gcmc_grid_logged = false;
  bool gcmc_parallel = false;

  GPU_Vector<int> cl_cell_count;
  GPU_Vector<int> cl_cell_count_sum;
  GPU_Vector<int> cl_cell_contents;

  double compute_isolated_atom_energy(int type);
  void gcmc_phase(int md_step, double temperature, Atom& atom, Box& box);
  void gcmc_phase_parallel(int md_step, double temperature, Atom& atom, Box& box);
  void write_statistics(
    int md_step,
    const std::vector<int>& htype,
    const std::vector<int>& num_ins_attempt,
    const std::vector<int>& num_ins_accept,
    const std::vector<int>& num_del_attempt,
    const std::vector<int>& num_del_accept,
    const std::vector<int>& num_ins_overlap);
};
