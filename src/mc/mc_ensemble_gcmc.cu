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

/*----------------------------------------------------------------------------80
The grand canonical ensemble for MCMD. Atoms of the listed species are inserted
into and deleted from spare atom slots (type -1) that the GCMC driver adds
before each call and removes afterwards.

Major contributors:
  Shinzato: Initial implementation.
------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "mc_ensemble_gcmc.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

static constexpr double LAMBDA_PREFACTOR = 17.45842;
static constexpr int GHOST_TYPE = -1;

static void compute_gcmc_domain_grid(
  const double thickness[3],
  double rc,
  int domain_separation_factor,
  int n_cell[3],
  int period[3],
  int s_vec[3])
{
  for (int d = 0; d < 3; ++d) {
    int n = (int)floor(thickness[d] / rc);
    if (n < 1) {
      n = 1;
    }
    double cell_len = thickness[d] / n;
    int m = (int)ceil(domain_separation_factor * rc / cell_len);
    if (m > n - 1) {
      m = n - 1;
    }
    if (m < 0) {
      m = 0;
    }
    n_cell[d] = n;
    period[d] = m + 1;
    s_vec[d] = n / (m + 1);
  }
}

static std::string get_gcmc_column_description(const std::vector<std::string>& tokens)
{
  std::string description = "num_MD_steps  acceptance_ratio  num_trials  [N  "
                            "insertion_acceptance_ratio  deletion_acceptance_ratio";
  if (std::find(tokens.begin(), tokens.end(), "overlap_cutoff") != tokens.end()) {
    description += "  num_overlap_rejections";
  }
  return description + " for each species]";
}

MC_Ensemble_GCMC::MC_Ensemble_GCMC(
  const std::vector<std::string>& tokens,
  int num_steps_mc_input,
  const std::string& potential_file_name)
  : MC_Ensemble(tokens, potential_file_name, get_gcmc_column_description(tokens))
{
  num_steps_mc = num_steps_mc_input;
  NN_ij.resize(1);
  NL_ij.resize(1000);
  candidate_min_d2.resize(1);
  const int n_max = 1000;
  const int m_max = 1000;
  NN_radial_after.resize(n_max);
  NN_angular_after.resize(n_max);
  x12_radial_after.resize(n_max * m_max);
  y12_radial_after.resize(n_max * m_max);
  z12_radial_after.resize(n_max * m_max);
  x12_angular_after.resize(n_max * m_max);
  y12_angular_after.resize(n_max * m_max);
  z12_angular_after.resize(n_max * m_max);
}

MC_Ensemble_GCMC::~MC_Ensemble_GCMC(void) { mc_output.close(); }

void MC_Ensemble_GCMC::set_seed(unsigned seed) { rng = std::mt19937(seed); }

static __global__ void
find_local_types(const int N_local, const int* atom_local, const int* g_type, int* g_local_type)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N_local) {
    g_local_type[k] = g_type[atom_local[k]];
  }
}

void MC_Ensemble_GCMC::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& groups,
  int grouping_method,
  int group_id)
{
  if (check_if_small_box(nep_energy.paramb.rc_radial_max, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }

  std::vector<char> deletable(atom.number_of_atoms, 1);
  if (grouping_method >= 0) {
    const std::vector<int>& label = groups[grouping_method].cpu_label;
    for (int n = 0; n < (int)label.size(); ++n) {
      deletable[n] = (label[n] == group_id);
    }
  }
  gcmc_phase(md_step, temperature, atom, box, deletable);
}

static __global__ void min_dist2_to_active(
  const int N,
  const Box box,
  const int exclude,
  const double qx,
  const double qy,
  const double qz,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_type,
  unsigned int* g_min_d2)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N && n != exclude && g_type[n] >= 0) {
    double dx = qx - g_x[n];
    double dy = qy - g_y[n];
    double dz = qz - g_z[n];
    apply_mic(box, dx, dy, dz);
    float d2 = float(dx * dx + dy * dy + dz * dz);
    atomicMin(g_min_d2, __float_as_uint(d2));
  }
}

static __global__ void get_neighbors_of_point(
  const int capacity,
  const Box box,
  const int site,
  const float rc_radial_square,
  const int exclude,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_type,
  int* g_NN,
  int* g_NL,
  const int NL_capacity)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < capacity) {
    if (g_type[n] < 0 || n == exclude || n == site) {
      return;
    }
    double x12 = g_x[site] - g_x[n];
    double y12 = g_y[site] - g_y[n];
    double z12 = g_z[site] - g_z[n];
    apply_mic(box, x12, y12, z12);
    float d2 = float(x12 * x12 + y12 * y12 + z12 * z12);
    if (d2 < rc_radial_square) {
      int slot = atomicAdd(g_NN, 1);
      if (slot < NL_capacity) {
        g_NL[slot] = n;
      }
    }
  }
}

static __global__ void create_inputs_gcmc(
  const int capacity,
  const int N_local,
  const int* atom_local,
  const Box box,
  const float rc_radial_square,
  const float rc_angular_square,
  const int exclude,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* g_type,
  int* g_NN_radial,
  int* g_NN_angular,
  int* g_t2_radial,
  int* g_t2_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n2 < capacity) {
    if (g_type[n2] < 0 || n2 == exclude) {
      return;
    }
    double x2 = g_x[n2];
    double y2 = g_y[n2];
    double z2 = g_z[n2];

    for (int k = 0; k < N_local; ++k) {
      int n1 = atom_local[k];
      if (n1 == n2) {
        continue;
      }
      double x12 = x2 - g_x[n1];
      double y12 = y2 - g_y[n1];
      double z12 = z2 - g_z[n1];
      apply_mic(box, x12, y12, z12);
      float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
      if (distance_square < rc_radial_square) {
        int count_radial = atomicAdd(&g_NN_radial[k], 1);
        int index_radial = count_radial * N_local + k;
        g_t2_radial[index_radial] = g_type[n2];
        g_x12_radial[index_radial] = float(x12);
        g_y12_radial[index_radial] = float(y12);
        g_z12_radial[index_radial] = float(z12);
      }
      if (distance_square < rc_angular_square) {
        int count_angular = atomicAdd(&g_NN_angular[k], 1);
        int index_angular = count_angular * N_local + k;
        g_t2_angular[index_angular] = g_type[n2];
        g_x12_angular[index_angular] = float(x12);
        g_y12_angular[index_angular] = float(y12);
        g_z12_angular[index_angular] = float(z12);
      }
    }
  }
}

template <typename T>
static void repack_host(std::vector<T>& v, int ncomp, int N0, int cap, const T& fill)
{
  if (v.size() != (size_t)ncomp * N0) {
    return;
  }
  std::vector<T> v2((size_t)cap * ncomp, fill);
  for (int c = 0; c < ncomp; ++c) {
    for (int i = 0; i < N0; ++i) {
      v2[(size_t)c * cap + i] = v[(size_t)c * N0 + i];
    }
  }
  v.swap(v2);
}

template <typename T>
static void repack_device(GPU_Vector<T>& v, int ncomp, int N0, int cap, const T& fill)
{
  if (v.size() != (size_t)ncomp * N0) {
    return;
  }
  std::vector<T> h(v.size());
  v.copy_to_host(h.data());
  repack_host(h, ncomp, N0, cap, fill);
  v.resize((size_t)cap * ncomp);
  v.copy_from_host(h.data());
}

void MC_Ensemble_GCMC::set_gcmc_parameters(
  const std::vector<std::string>& gcmc_species_input,
  const std::vector<int>& gcmc_types_input,
  const std::vector<double>& gcmc_mu_input,
  const std::vector<double>& gcmc_mass_input,
  const std::vector<double>& gcmc_e0_input,
  const std::vector<int>& gcmc_e0_auto_input,
  int num_gcmc_sweeps_input,
  double gcmc_insertion_probability_input,
  int domain_separation_factor_input,
  bool gcmc_parallel_input)
{
  gcmc_species = gcmc_species_input;
  gcmc_types = gcmc_types_input;
  gcmc_mu = gcmc_mu_input;
  gcmc_mass = gcmc_mass_input;
  gcmc_e0 = gcmc_e0_input;
  num_gcmc_sweeps = num_gcmc_sweeps_input;
  gcmc_insertion_probability = gcmc_insertion_probability_input;
  domain_separation_factor = domain_separation_factor_input;
  gcmc_parallel = gcmc_parallel_input;

  for (int s = 0; s < (int)gcmc_species.size(); ++s) {
    if (s < (int)gcmc_e0_auto_input.size() && gcmc_e0_auto_input[s]) {
      gcmc_e0[s] = compute_isolated_atom_energy(gcmc_types[s]);
      printf(
        "        GCMC species %s: auto isolated-atom energy e0 = %g eV.\n",
        gcmc_species[s].c_str(),
        gcmc_e0[s]);
    }
  }
}

void MC_Ensemble_GCMC::set_overlap_cutoffs(const std::vector<float>& overlap_cutoff2_by_type_input)
{
  const float rc = nep_energy.paramb.rc_radial_max;
  const float rc2 = rc * rc;
  for (size_t t = 0; t < overlap_cutoff2_by_type_input.size(); ++t) {
    if (overlap_cutoff2_by_type_input[t] > rc2) {
      PRINT_INPUT_ERROR("overlap_cutoff for mc gcmc should not exceed the NEP radial cutoff.\n");
    }
  }
  overlap_cutoff2_by_type = overlap_cutoff2_by_type_input;
}

double MC_Ensemble_GCMC::compute_isolated_atom_energy(int type)
{
  GPU_Vector<int> NN_radial_one(1), NN_angular_one(1), tp(1), t2_radial(1), t2_angular(1);
  GPU_Vector<float> x12_radial_one(1), y12_radial_one(1), z12_radial_one(1);
  GPU_Vector<float> x12_angular_one(1), y12_angular_one(1), z12_angular_one(1);
  GPU_Vector<float> pe(1);
  int zero = 0;
  NN_radial_one.copy_from_host(&zero);
  NN_angular_one.copy_from_host(&zero);
  tp.copy_from_host(&type);
  nep_energy.find_energy(
    1,
    NN_radial_one.data(),
    NN_angular_one.data(),
    tp.data(),
    t2_radial.data(),
    t2_angular.data(),
    x12_radial_one.data(),
    y12_radial_one.data(),
    z12_radial_one.data(),
    x12_angular_one.data(),
    y12_angular_one.data(),
    z12_angular_one.data(),
    pe.data());
  float e = 0.0f;
  pe.copy_to_host(&e);
  return (double)e + nep_energy.energy_reference[type];
}

int MC_Ensemble_GCMC::get_max_insertions(Box& box)
{
  double volume = box.get_volume();
  double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  int n_cell[3], period[3], s_vec[3];
  compute_gcmc_domain_grid(
    thickness, nep_energy.paramb.rc_radial_max, domain_separation_factor, n_cell, period, s_vec);
  return num_gcmc_sweeps * s_vec[0] * s_vec[1] * s_vec[2];
}

void MC_Ensemble_GCMC::add_ghost_slots(Atom& atom, Box& box, int num_slots)
{
  if (atom.has_spin) {
    PRINT_INPUT_ERROR("mc gcmc does not support spin models.\n");
  }
  const int N0 = atom.number_of_atoms;
  const int cap = N0 + num_slots;

  repack_device(atom.type, 1, N0, cap, GHOST_TYPE);
  repack_device(atom.mass, 1, N0, cap, 1.0);
  repack_device(atom.charge, 1, N0, cap, 0.0f);
  repack_device(atom.position_per_atom, 3, N0, cap, 0.0);
  repack_device(atom.velocity_per_atom, 3, N0, cap, 0.0);

  repack_host(atom.cpu_type, 1, N0, cap, GHOST_TYPE);
  repack_host(atom.cpu_mass, 1, N0, cap, 1.0);
  repack_host(atom.cpu_charge, 1, N0, cap, 0.0f);
  repack_host(atom.cpu_position_per_atom, 3, N0, cap, 0.0);
  repack_host(atom.cpu_velocity_per_atom, 3, N0, cap, 0.0);
  repack_host(atom.cpu_atom_symbol, 1, N0, cap, std::string("X"));

  const double* h = box.cpu_h;
  std::uniform_real_distribution<double> uf(0.0, 1.0);
  std::vector<double> gx(num_slots), gy(num_slots), gz(num_slots);
  for (int k = 0; k < num_slots; ++k) {
    double f0 = uf(rng), f1 = uf(rng), f2 = uf(rng);
    gx[k] = h[0] * f0 + h[1] * f1 + h[2] * f2;
    gy[k] = h[3] * f0 + h[4] * f1 + h[5] * f2;
    gz[k] = h[6] * f0 + h[7] * f1 + h[8] * f2;
  }
  atom.position_per_atom.copy_from_host(gx.data(), num_slots, N0);
  atom.position_per_atom.copy_from_host(gy.data(), num_slots, cap + N0);
  atom.position_per_atom.copy_from_host(gz.data(), num_slots, 2 * cap + N0);

  atom.number_of_atoms = cap;
}

void MC_Ensemble_GCMC::write_statistics(
  int md_step,
  const std::vector<int>& htype,
  const std::vector<int>& num_ins_attempt,
  const std::vector<int>& num_ins_accept,
  const std::vector<int>& num_del_attempt,
  const std::vector<int>& num_del_accept,
  const std::vector<int>& num_overlap)
{
  std::vector<int> count(gcmc_types.size(), 0);
  for (int n = 0; n < (int)htype.size(); ++n) {
    for (int s = 0; s < (int)gcmc_types.size(); ++s) {
      if (htype[n] == gcmc_types[s]) {
        ++count[s];
      }
    }
  }
  int num_trials = 0;
  int num_accepted = 0;
  for (int s = 0; s < (int)gcmc_types.size(); ++s) {
    num_trials += num_ins_attempt[s] + num_del_attempt[s];
    num_accepted += num_ins_accept[s] + num_del_accept[s];
  }
  auto ratio = [](int accepted, int attempted) {
    return attempted > 0 ? accepted / double(attempted) : 0.0;
  };
  mc_output << md_step << "  " << ratio(num_accepted, num_trials) << " " << num_trials;
  for (int s = 0; s < (int)gcmc_types.size(); ++s) {
    mc_output << " " << count[s] << " " << ratio(num_ins_accept[s], num_ins_attempt[s]) << " "
              << ratio(num_del_accept[s], num_del_attempt[s]);
    if (!overlap_cutoff2_by_type.empty()) {
      mc_output << " " << num_overlap[s];
    }
  }
  mc_output << std::endl;
}

void MC_Ensemble_GCMC::gcmc_phase(
  int md_step, double temperature, Atom& atom, Box& box, const std::vector<char>& deletable)
{
  if (gcmc_species.empty() || num_gcmc_sweeps <= 0) {
    return;
  }
  const int cap = atom.number_of_atoms;

  if (gcmc_parallel) {
    int num_bins_probe[3];
    bool too_small_for_cell_list =
      box.get_num_bins(0.5 * nep_energy.paramb.rc_radial_max, num_bins_probe);
    if (!too_small_for_cell_list) {
      gcmc_phase_parallel(md_step, temperature, atom, box, deletable);
      return;
    }
  }

  box.get_inverse();
  const double* h = box.cpu_h;
  const double* hinv = box.cpu_h + 9;
  double volume = box.get_volume();
  double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  double rc = nep_energy.paramb.rc_radial_max;
  const float rc_radial_square = rc * rc;
  const float rc_angular_square =
    nep_energy.paramb.rc_angular_max * nep_energy.paramb.rc_angular_max;

  int n_cell[3], period[3], s_vec[3];
  compute_gcmc_domain_grid(thickness, rc, domain_separation_factor, n_cell, period, s_vec);
  int n_total = n_cell[0] * n_cell[1] * n_cell[2];
  double V_cell = volume / n_total;
  double beta = 1.0 / (K_B * temperature);
  const double ratio_ins = (1.0 - gcmc_insertion_probability) / gcmc_insertion_probability;
  const double ratio_del = gcmc_insertion_probability / (1.0 - gcmc_insertion_probability);

  double* g_x = atom.position_per_atom.data();
  double* g_y = g_x + cap;
  double* g_z = g_x + 2 * cap;
  const int blocks = (cap - 1) / 64 + 1;
  const int NL_cap = (int)NL_ij.size();

  auto present_absent = [&](int site, double& E_present, double& E_absent) -> bool {
    CHECK(gpuMemset(NN_ij.data(), 0, sizeof(int)));
    get_neighbors_of_point<<<blocks, 64>>>(
      cap,
      box,
      site,
      rc_radial_square,
      site,
      g_x,
      g_y,
      g_z,
      atom.type.data(),
      NN_ij.data(),
      NL_ij.data(),
      NL_cap);
    GPU_CHECK_KERNEL
    int NN0;
    NN_ij.copy_to_host(&NN0);
    if (NN0 + 1 > (int)NN_radial_after.size() || NN0 + 1 > NL_cap) {
      return false;
    }

    find_local_types<<<(NN0 == 0 ? 1 : (NN0 - 1) / 64 + 1), 64>>>(
      NN0, NL_ij.data(), atom.type.data(), local_type_before.data());
    GPU_CHECK_KERNEL
    CHECK(gpuMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(gpuMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));
    create_inputs_gcmc<<<blocks, 64>>>(
      cap,
      NN0,
      NL_ij.data(),
      box,
      rc_radial_square,
      rc_angular_square,
      site,
      g_x,
      g_y,
      g_z,
      atom.type.data(),
      NN_radial.data(),
      NN_angular.data(),
      t2_radial_before.data(),
      t2_angular_before.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data());
    GPU_CHECK_KERNEL
    nep_energy.find_energy(
      NN0,
      NN_radial.data(),
      NN_angular.data(),
      local_type_before.data(),
      t2_radial_before.data(),
      t2_angular_before.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_before.data());

    NL_ij.copy_from_host(&site, 1, NN0);
    int NN1 = NN0 + 1;
    find_local_types<<<(NN1 - 1) / 64 + 1, 64>>>(
      NN1, NL_ij.data(), atom.type.data(), local_type_before.data());
    GPU_CHECK_KERNEL
    CHECK(gpuMemset(NN_radial_after.data(), 0, sizeof(int) * NN_radial_after.size()));
    CHECK(gpuMemset(NN_angular_after.data(), 0, sizeof(int) * NN_angular_after.size()));
    create_inputs_gcmc<<<blocks, 64>>>(
      cap,
      NN1,
      NL_ij.data(),
      box,
      rc_radial_square,
      rc_angular_square,
      -1,
      g_x,
      g_y,
      g_z,
      atom.type.data(),
      NN_radial_after.data(),
      NN_angular_after.data(),
      t2_radial_after.data(),
      t2_angular_after.data(),
      x12_radial_after.data(),
      y12_radial_after.data(),
      z12_radial_after.data(),
      x12_angular_after.data(),
      y12_angular_after.data(),
      z12_angular_after.data());
    GPU_CHECK_KERNEL
    nep_energy.find_energy(
      NN1,
      NN_radial_after.data(),
      NN_angular_after.data(),
      local_type_before.data(),
      t2_radial_after.data(),
      t2_angular_after.data(),
      x12_radial_after.data(),
      y12_radial_after.data(),
      z12_radial_after.data(),
      x12_angular_after.data(),
      y12_angular_after.data(),
      z12_angular_after.data(),
      pe_after.data());

    std::vector<float> pe_abs(NN0), pe_pre(NN1);
    if (NN0 > 0) {
      pe_before.copy_to_host(pe_abs.data(), NN0);
    }
    pe_after.copy_to_host(pe_pre.data(), NN1);
    E_absent = 0.0;
    for (int n = 0; n < NN0; ++n) {
      E_absent += pe_abs[n];
    }
    E_present = 0.0;
    for (int n = 0; n < NN1; ++n) {
      E_present += pe_pre[n];
    }
    return true;
  };

  auto cell_of = [&](double x, double y, double z) -> int {
    double s[3] = {
      hinv[0] * x + hinv[1] * y + hinv[2] * z,
      hinv[3] * x + hinv[4] * y + hinv[5] * z,
      hinv[6] * x + hinv[7] * y + hinv[8] * z};
    int c[3];
    for (int d = 0; d < 3; ++d) {
      double f = s[d] - floor(s[d]);
      c[d] = (int)(f * n_cell[d]);
      if (c[d] >= n_cell[d]) {
        c[d] = n_cell[d] - 1;
      }
    }
    return (c[0] * n_cell[1] + c[1]) * n_cell[2] + c[2];
  };

  std::vector<int> htype(cap);
  std::vector<double> hpos(3 * cap);
  atom.type.copy_to_host(htype.data());
  atom.position_per_atom.copy_to_host(hpos.data());

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<int> num_ins_attempt(gcmc_species.size(), 0);
  std::vector<int> num_ins_accept(gcmc_species.size(), 0);
  std::vector<int> num_del_attempt(gcmc_species.size(), 0);
  std::vector<int> num_del_accept(gcmc_species.size(), 0);
  std::vector<int> num_overlap(gcmc_species.size(), 0);
  const bool overlap_cutoff_active = !overlap_cutoff2_by_type.empty();

  auto has_overlap = [&](int site, double x, double y, double z, int type) -> bool {
    if (
      !overlap_cutoff_active || type >= (int)overlap_cutoff2_by_type.size() ||
      overlap_cutoff2_by_type[type] <= 0.0f) {
      return false;
    }
    CHECK(gpuMemset(candidate_min_d2.data(), 0x7f, sizeof(unsigned int)));
    min_dist2_to_active<<<blocks, 64>>>(
      cap, box, site, x, y, z, g_x, g_y, g_z, atom.type.data(), candidate_min_d2.data());
    GPU_CHECK_KERNEL
    unsigned int md2_bits;
    candidate_min_d2.copy_to_host(&md2_bits, 1);
    float md2;
    std::memcpy(&md2, &md2_bits, sizeof(float));
    return md2 < overlap_cutoff2_by_type[type];
  };

  for (int sweep = 0; sweep < num_gcmc_sweeps; ++sweep) {
    std::vector<std::vector<int>> cell_atoms(n_total);
    std::vector<int> free_slots;
    for (int n = 0; n < cap; ++n) {
      int t = htype[n];
      if (t < 0) {
        free_slots.push_back(n);
      } else {
        bool is_impurity = false;
        for (int s = 0; s < (int)gcmc_types.size(); ++s) {
          if (t == gcmc_types[s]) {
            is_impurity = true;
            break;
          }
        }
        if (is_impurity && deletable[n]) {
          cell_atoms[cell_of(hpos[n], hpos[cap + n], hpos[2 * cap + n])].push_back(n);
        }
      }
    }

    int o[3] = {
      (int)(uni(rng) * n_cell[0]), (int)(uni(rng) * n_cell[1]), (int)(uni(rng) * n_cell[2])};
    for (int p0 = 0; p0 < n_cell[0]; ++p0) {
      for (int p1 = 0; p1 < n_cell[1]; ++p1) {
        for (int p2 = 0; p2 < n_cell[2]; ++p2) {
          int p[3] = {p0, p1, p2};
          bool active = true;
          for (int d = 0; d < 3; ++d) {
            int dd = (p[d] + n_cell[d] - o[d]) % n_cell[d];
            if (dd % period[d] != 0 || dd >= s_vec[d] * period[d]) {
              active = false;
              break;
            }
          }
          if (!active) {
            continue;
          }
          int cell_id = (p0 * n_cell[1] + p1) * n_cell[2] + p2;

          int s_idx = (int)(uni(rng) * gcmc_species.size());
          if (s_idx >= (int)gcmc_species.size()) {
            s_idx = (int)gcmc_species.size() - 1;
          }
          int t_ins = gcmc_types[s_idx];
          double mu = gcmc_mu[s_idx];
          double e0 = gcmc_e0[s_idx];
          const double e_ref = nep_energy.energy_reference[t_ins];
          double lambda = LAMBDA_PREFACTOR / sqrt(gcmc_mass[s_idx] * temperature);
          double lambda3 = lambda * lambda * lambda;

          int N_s = 0;
          for (int idx : cell_atoms[cell_id]) {
            if (htype[idx] == t_ins) {
              ++N_s;
            }
          }

          if (uni(rng) < gcmc_insertion_probability) {
            ++num_ins_attempt[s_idx];
            if (free_slots.empty()) {
              continue;
            }
            int g = free_slots.back();
            double fc[3] = {
              (p0 + uni(rng)) / n_cell[0],
              (p1 + uni(rng)) / n_cell[1],
              (p2 + uni(rng)) / n_cell[2]};
            double Bx = h[0] * fc[0] + h[1] * fc[1] + h[2] * fc[2];
            double By = h[3] * fc[0] + h[4] * fc[1] + h[5] * fc[2];
            double Bz = h[6] * fc[0] + h[7] * fc[1] + h[8] * fc[2];
            atom.position_per_atom.copy_from_host(&Bx, 1, g);
            atom.position_per_atom.copy_from_host(&By, 1, cap + g);
            atom.position_per_atom.copy_from_host(&Bz, 1, 2 * cap + g);
            atom.type.copy_from_host(&t_ins, 1, g);

            if (has_overlap(g, Bx, By, Bz, t_ins)) {
              int ghost = GHOST_TYPE;
              atom.type.copy_from_host(&ghost, 1, g);
              ++num_overlap[s_idx];
              continue;
            }

            double E_present, E_absent;
            if (!present_absent(g, E_present, E_absent)) {
              int ghost = GHOST_TYPE;
              atom.type.copy_from_host(&ghost, 1, g);
              continue;
            }
            double dU = E_present - E_absent + e_ref;
            double prefactor = ratio_ins * V_cell / (lambda3 * (N_s + 1));
            double boltzmann = exp(beta * (mu - (dU - e0)));
            double acc = prefactor * boltzmann;
            bool accepted = (uni(rng) < acc);
            if (accepted) {
              ++num_ins_accept[s_idx];
              double mass = gcmc_mass[s_idx];
              double sigma = sqrt(K_B * temperature / mass);
              std::normal_distribution<double> maxwell(0.0, sigma);
              double vx_ins = maxwell(rng);
              double vy_ins = maxwell(rng);
              double vz_ins = maxwell(rng);
              atom.mass.copy_from_host(&mass, 1, g);
              atom.velocity_per_atom.copy_from_host(&vx_ins, 1, g);
              atom.velocity_per_atom.copy_from_host(&vy_ins, 1, cap + g);
              atom.velocity_per_atom.copy_from_host(&vz_ins, 1, 2 * cap + g);
              atom.cpu_type[g] = t_ins;
              atom.cpu_mass[g] = mass;
              atom.cpu_atom_symbol[g] = gcmc_species[s_idx];
              htype[g] = t_ins;
              hpos[g] = Bx;
              hpos[cap + g] = By;
              hpos[2 * cap + g] = Bz;
              free_slots.pop_back();
              cell_atoms[cell_id].push_back(g);
            } else {
              int ghost = GHOST_TYPE;
              atom.type.copy_from_host(&ghost, 1, g);
            }
          } else {
            ++num_del_attempt[s_idx];
            if (N_s == 0) {
              continue;
            }
            std::vector<int> candidates;
            for (int idx : cell_atoms[cell_id]) {
              if (htype[idx] == t_ins) {
                candidates.push_back(idx);
              }
            }
            int j = candidates[(int)(uni(rng) * candidates.size()) % candidates.size()];
            if (has_overlap(j, hpos[j], hpos[cap + j], hpos[2 * cap + j], t_ins)) {
              ++num_overlap[s_idx];
              continue;
            }

            double E_present, E_absent;
            if (!present_absent(j, E_present, E_absent)) {
              continue;
            }
            double dU = E_absent - E_present - e_ref;
            double prefactor = ratio_del * lambda3 * (double)N_s / V_cell;
            double boltzmann = exp(-beta * (mu + (dU + e0)));
            double acc = prefactor * boltzmann;
            bool accepted = (uni(rng) < acc);
            if (accepted) {
              ++num_del_accept[s_idx];
              int ghost = GHOST_TYPE;
              atom.type.copy_from_host(&ghost, 1, j);
              atom.cpu_type[j] = GHOST_TYPE;
              atom.cpu_atom_symbol[j] = std::string("X");
              htype[j] = GHOST_TYPE;
              double zero = 0.0;
              atom.velocity_per_atom.copy_from_host(&zero, 1, j);
              atom.velocity_per_atom.copy_from_host(&zero, 1, cap + j);
              atom.velocity_per_atom.copy_from_host(&zero, 1, 2 * cap + j);
              free_slots.push_back(j);
              auto& lst = cell_atoms[cell_id];
              for (size_t q = 0; q < lst.size(); ++q) {
                if (lst[q] == j) {
                  lst[q] = lst.back();
                  lst.pop_back();
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  write_statistics(
    md_step, htype, num_ins_attempt, num_ins_accept, num_del_attempt, num_del_accept, num_overlap);
}

static __global__ void gcmc_par_find_site_neighbors(
  const Box box,
  const int M,
  const int* __restrict__ sites,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const int max_nb,
  int* g_site_NN,
  int* g_site_NL)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) {
    return;
  }
  const int site = sites[m];
  const double x1 = g_x[site];
  const double y1 = g_y[site];
  const double z1 = g_z[site];
  int cx, cy, cz, cid;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cx, cy, cz, cid);
  int z_begin, z_end, y_begin, y_end, x_begin, x_end;
  get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
  get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
  get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);
  int count = 0;
  for (int k = z_begin; k < z_end; ++k) {
    for (int j = y_begin; j < y_end; ++j) {
      for (int i = x_begin; i < x_end; ++i) {
        int neighbor_cell = cid + k * nx * ny + j * nx + i;
        if (cx + i < 0)
          neighbor_cell += nx;
        if (cx + i >= nx)
          neighbor_cell -= nx;
        if (cy + j < 0)
          neighbor_cell += ny * nx;
        if (cy + j >= ny)
          neighbor_cell -= ny * nx;
        if (cz + k < 0)
          neighbor_cell += nz * ny * nx;
        if (cz + k >= nz)
          neighbor_cell -= nz * ny * nx;
        const int num = cell_counts[neighbor_cell];
        const int prev = cell_count_sum[neighbor_cell];
        for (int t = 0; t < num; ++t) {
          const int n2 = cell_contents[prev + t];
          if (n2 == site || g_type[n2] < 0) {
            continue;
          }
          double x12 = g_x[n2] - x1;
          double y12 = g_y[n2] - y1;
          double z12 = g_z[n2] - z1;
          apply_mic(box, x12, y12, z12);
          float d2 = float(x12 * x12 + y12 * y12 + z12 * z12);
          if (d2 < rc_radial_square) {
            if (count < max_nb) {
              g_site_NL[m * max_nb + count] = n2;
            }
            ++count;
          }
        }
      }
    }
  }
  g_site_NN[m] = count;
}

static __global__ void gcmc_par_build_inputs(
  const Box box,
  const int N_c,
  const int* __restrict__ center_slot,
  const int* __restrict__ center_exclude,
  const float rc_radial_square,
  const float rc_angular_square,
  const int MN_radial,
  const int MN_angular,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  int* g_center_type,
  int* g_NN_radial,
  int* g_NN_angular,
  int* g_t2_radial,
  int* g_t2_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= N_c) {
    return;
  }
  const int n1 = center_slot[c];
  const int excl = center_exclude[c];
  g_center_type[c] = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  int cx, cy, cz, cid;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cx, cy, cz, cid);
  int z_begin, z_end, y_begin, y_end, x_begin, x_end;
  get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
  get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
  get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);
  int nn_r = 0, nn_a = 0;
  for (int k = z_begin; k < z_end; ++k) {
    for (int j = y_begin; j < y_end; ++j) {
      for (int i = x_begin; i < x_end; ++i) {
        int neighbor_cell = cid + k * nx * ny + j * nx + i;
        if (cx + i < 0)
          neighbor_cell += nx;
        if (cx + i >= nx)
          neighbor_cell -= nx;
        if (cy + j < 0)
          neighbor_cell += ny * nx;
        if (cy + j >= ny)
          neighbor_cell -= ny * nx;
        if (cz + k < 0)
          neighbor_cell += nz * ny * nx;
        if (cz + k >= nz)
          neighbor_cell -= nz * ny * nx;
        const int num = cell_counts[neighbor_cell];
        const int prev = cell_count_sum[neighbor_cell];
        for (int t = 0; t < num; ++t) {
          const int n2 = cell_contents[prev + t];
          if (n2 == n1 || n2 == excl || g_type[n2] < 0) {
            continue;
          }
          double x12 = g_x[n2] - x1;
          double y12 = g_y[n2] - y1;
          double z12 = g_z[n2] - z1;
          apply_mic(box, x12, y12, z12);
          float d2 = float(x12 * x12 + y12 * y12 + z12 * z12);
          if (d2 < rc_radial_square && nn_r < MN_radial) {
            int idx = nn_r * N_c + c;
            g_t2_radial[idx] = g_type[n2];
            g_x12_radial[idx] = float(x12);
            g_y12_radial[idx] = float(y12);
            g_z12_radial[idx] = float(z12);
            ++nn_r;
          }
          if (d2 < rc_angular_square && nn_a < MN_angular) {
            int idx = nn_a * N_c + c;
            g_t2_angular[idx] = g_type[n2];
            g_x12_angular[idx] = float(x12);
            g_y12_angular[idx] = float(y12);
            g_z12_angular[idx] = float(z12);
            ++nn_a;
          }
        }
      }
    }
  }
  g_NN_radial[c] = nn_r;
  g_NN_angular[c] = nn_a;
}

void MC_Ensemble_GCMC::gcmc_phase_parallel(
  int md_step, double temperature, Atom& atom, Box& box, const std::vector<char>& deletable)
{
  const int cap = atom.number_of_atoms;
  box.get_inverse();
  const double* h = box.cpu_h;
  const double* hinv = box.cpu_h + 9;
  double volume = box.get_volume();
  double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  double rc = nep_energy.paramb.rc_radial_max;
  const float rc_radial_square = rc * rc;
  const float rc_angular_square =
    nep_energy.paramb.rc_angular_max * nep_energy.paramb.rc_angular_max;
  const int MN_r = nep_energy.paramb.MN_radial;
  const int MN_a = nep_energy.paramb.MN_angular;
  const int max_nb = MN_r;

  int n_cell[3], period[3], s_vec[3];
  compute_gcmc_domain_grid(thickness, rc, domain_separation_factor, n_cell, period, s_vec);
  if (!gcmc_grid_logged) {
    gcmc_grid_logged = true;
    printf(
      "    GCMC parallel grid: n_cell = %d x %d x %d, period = %d x %d x %d, "
      "active cells = %d.\n",
      n_cell[0],
      n_cell[1],
      n_cell[2],
      period[0],
      period[1],
      period[2],
      s_vec[0] * s_vec[1] * s_vec[2]);
  }
  int n_total = n_cell[0] * n_cell[1] * n_cell[2];
  double V_cell = volume / n_total;
  double beta = 1.0 / (K_B * temperature);
  const double ratio_ins = (1.0 - gcmc_insertion_probability) / gcmc_insertion_probability;
  const double ratio_del = gcmc_insertion_probability / (1.0 - gcmc_insertion_probability);

  const double rc_cl = 0.5 * rc;
  const double rc_cl_inv = 1.0 / rc_cl;
  int num_bins[3];
  box.get_num_bins(rc_cl, num_bins);
  if ((int)cl_cell_contents.size() < cap) {
    cl_cell_contents.resize(cap);
  }

  double* g_x = atom.position_per_atom.data();

  std::vector<int> htype(cap);
  std::vector<double> hpos(3 * cap);
  atom.type.copy_to_host(htype.data());
  atom.position_per_atom.copy_to_host(hpos.data());

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<int> num_ins_attempt(gcmc_species.size(), 0);
  std::vector<int> num_ins_accept(gcmc_species.size(), 0);
  std::vector<int> num_del_attempt(gcmc_species.size(), 0);
  std::vector<int> num_del_accept(gcmc_species.size(), 0);
  std::vector<int> num_overlap(gcmc_species.size(), 0);
  const bool overlap_cutoff_active = !overlap_cutoff2_by_type.empty();

  auto min_image_d2 =
    [&](double ax, double ay, double az, double bx, double by, double bz) -> double {
    double dx = ax - bx, dy = ay - by, dz = az - bz;
    double sx = hinv[0] * dx + hinv[1] * dy + hinv[2] * dz;
    double sy = hinv[3] * dx + hinv[4] * dy + hinv[5] * dz;
    double sz = hinv[6] * dx + hinv[7] * dy + hinv[8] * dz;
    if (box.pbc_x) {
      sx -= round(sx);
    }
    if (box.pbc_y) {
      sy -= round(sy);
    }
    if (box.pbc_z) {
      sz -= round(sz);
    }
    double rx = h[0] * sx + h[1] * sy + h[2] * sz;
    double ry = h[3] * sx + h[4] * sy + h[5] * sz;
    double rz = h[6] * sx + h[7] * sy + h[8] * sz;
    return rx * rx + ry * ry + rz * rz;
  };

  auto cell_of = [&](double x, double y, double z) -> int {
    double s[3] = {
      hinv[0] * x + hinv[1] * y + hinv[2] * z,
      hinv[3] * x + hinv[4] * y + hinv[5] * z,
      hinv[6] * x + hinv[7] * y + hinv[8] * z};
    int c[3];
    for (int d = 0; d < 3; ++d) {
      double f = s[d] - floor(s[d]);
      c[d] = (int)(f * n_cell[d]);
      if (c[d] >= n_cell[d]) {
        c[d] = n_cell[d] - 1;
      }
    }
    return (c[0] * n_cell[1] + c[1]) * n_cell[2] + c[2];
  };

  GPU_Vector<int> d_sites, d_site_NN, d_site_NL;
  GPU_Vector<int> d_slot, d_excl, d_ctype, d_NN_r, d_NN_a, d_t2_r, d_t2_a;
  GPU_Vector<float> d_x12r, d_y12r, d_z12r, d_x12a, d_y12a, d_z12a, d_pe;

  auto eval_centers =
    [&](const std::vector<int>& slot, const std::vector<int>& excl, std::vector<float>& pe_host) {
      int N_c = (int)slot.size();
      pe_host.assign(N_c, 0.0f);
      if (N_c == 0) {
        return;
      }
      if (d_slot.size() < (size_t)N_c) {
        d_slot.resize(N_c);
        d_excl.resize(N_c);
        d_ctype.resize(N_c);
        d_NN_r.resize(N_c);
        d_NN_a.resize(N_c);
        d_pe.resize(N_c);
      }
      if (d_t2_r.size() < (size_t)N_c * MN_r) {
        d_t2_r.resize((size_t)N_c * MN_r);
        d_x12r.resize((size_t)N_c * MN_r);
        d_y12r.resize((size_t)N_c * MN_r);
        d_z12r.resize((size_t)N_c * MN_r);
      }
      if (d_t2_a.size() < (size_t)N_c * MN_a) {
        d_t2_a.resize((size_t)N_c * MN_a);
        d_x12a.resize((size_t)N_c * MN_a);
        d_y12a.resize((size_t)N_c * MN_a);
        d_z12a.resize((size_t)N_c * MN_a);
      }
      d_slot.copy_from_host(slot.data(), N_c);
      d_excl.copy_from_host(excl.data(), N_c);
      int blk = (N_c - 1) / 64 + 1;
      gcmc_par_build_inputs<<<blk, 64>>>(
        box,
        N_c,
        d_slot.data(),
        d_excl.data(),
        rc_radial_square,
        rc_angular_square,
        MN_r,
        MN_a,
        g_x,
        g_x + cap,
        g_x + 2 * cap,
        atom.type.data(),
        cl_cell_count.data(),
        cl_cell_count_sum.data(),
        cl_cell_contents.data(),
        num_bins[0],
        num_bins[1],
        num_bins[2],
        rc_cl_inv,
        d_ctype.data(),
        d_NN_r.data(),
        d_NN_a.data(),
        d_t2_r.data(),
        d_t2_a.data(),
        d_x12r.data(),
        d_y12r.data(),
        d_z12r.data(),
        d_x12a.data(),
        d_y12a.data(),
        d_z12a.data());
      GPU_CHECK_KERNEL
      nep_energy.find_energy(
        N_c,
        d_NN_r.data(),
        d_NN_a.data(),
        d_ctype.data(),
        d_t2_r.data(),
        d_t2_a.data(),
        d_x12r.data(),
        d_y12r.data(),
        d_z12r.data(),
        d_x12a.data(),
        d_y12a.data(),
        d_z12a.data(),
        d_pe.data());
      d_pe.copy_to_host(pe_host.data(), N_c);
    };

  enum { INSERT, DELETE };
  struct Trial {
    int kind;
    int site;
    int s_idx;
    int N_s;
  };

  for (int sweep = 0; sweep < num_gcmc_sweeps; ++sweep) {
    std::vector<std::vector<int>> cell_atoms(n_total);
    std::vector<int> free_slots;
    for (int n = 0; n < cap; ++n) {
      int t = htype[n];
      if (t < 0) {
        free_slots.push_back(n);
      } else {
        for (int s = 0; s < (int)gcmc_types.size(); ++s) {
          if (t == gcmc_types[s]) {
            if (deletable[n]) {
              cell_atoms[cell_of(hpos[n], hpos[cap + n], hpos[2 * cap + n])].push_back(n);
            }
            break;
          }
        }
      }
    }

    int o[3] = {
      (int)(uni(rng) * n_cell[0]), (int)(uni(rng) * n_cell[1]), (int)(uni(rng) * n_cell[2])};
    std::vector<Trial> trials;
    for (int p0 = 0; p0 < n_cell[0]; ++p0) {
      for (int p1 = 0; p1 < n_cell[1]; ++p1) {
        for (int p2 = 0; p2 < n_cell[2]; ++p2) {
          int p[3] = {p0, p1, p2};
          bool active = true;
          for (int d = 0; d < 3; ++d) {
            int dd = (p[d] + n_cell[d] - o[d]) % n_cell[d];
            if (dd % period[d] != 0 || dd >= s_vec[d] * period[d]) {
              active = false;
              break;
            }
          }
          if (!active) {
            continue;
          }
          int cell_id = (p0 * n_cell[1] + p1) * n_cell[2] + p2;
          int s_idx = (int)(uni(rng) * gcmc_species.size());
          if (s_idx >= (int)gcmc_species.size()) {
            s_idx = (int)gcmc_species.size() - 1;
          }
          int t_ins = gcmc_types[s_idx];
          int N_s = 0;
          for (int idx : cell_atoms[cell_id]) {
            if (htype[idx] == t_ins) {
              ++N_s;
            }
          }
          if (uni(rng) < gcmc_insertion_probability) {
            ++num_ins_attempt[s_idx];
            if (free_slots.empty()) {
              continue;
            }
            int g = free_slots.back();
            free_slots.pop_back();
            double fc[3] = {
              (p0 + uni(rng)) / n_cell[0],
              (p1 + uni(rng)) / n_cell[1],
              (p2 + uni(rng)) / n_cell[2]};
            double Bx = h[0] * fc[0] + h[1] * fc[1] + h[2] * fc[2];
            double By = h[3] * fc[0] + h[4] * fc[1] + h[5] * fc[2];
            double Bz = h[6] * fc[0] + h[7] * fc[1] + h[8] * fc[2];
            atom.position_per_atom.copy_from_host(&Bx, 1, g);
            atom.position_per_atom.copy_from_host(&By, 1, cap + g);
            atom.position_per_atom.copy_from_host(&Bz, 1, 2 * cap + g);
            atom.type.copy_from_host(&t_ins, 1, g);
            htype[g] = t_ins;
            hpos[g] = Bx;
            hpos[cap + g] = By;
            hpos[2 * cap + g] = Bz;
            trials.push_back({INSERT, g, s_idx, N_s});
          } else {
            ++num_del_attempt[s_idx];
            if (N_s == 0) {
              continue;
            }
            std::vector<int> cand;
            for (int idx : cell_atoms[cell_id]) {
              if (htype[idx] == t_ins) {
                cand.push_back(idx);
              }
            }
            int j = cand[(int)(uni(rng) * cand.size()) % cand.size()];
            trials.push_back({DELETE, j, s_idx, N_s});
          }
        }
      }
    }

    int M = (int)trials.size();
    if (M == 0) {
      continue;
    }

    find_cell_list(
      rc_cl,
      num_bins,
      box,
      atom.position_per_atom,
      cl_cell_count,
      cl_cell_count_sum,
      cl_cell_contents);

    if (d_sites.size() < (size_t)M) {
      d_sites.resize(M);
      d_site_NN.resize(M);
    }
    if (d_site_NL.size() < (size_t)M * max_nb) {
      d_site_NL.resize((size_t)M * max_nb);
    }
    std::vector<int> sites(M);
    for (int m = 0; m < M; ++m) {
      sites[m] = trials[m].site;
    }
    d_sites.copy_from_host(sites.data(), M);
    gcmc_par_find_site_neighbors<<<(M - 1) / 64 + 1, 64>>>(
      box,
      M,
      d_sites.data(),
      rc_radial_square,
      g_x,
      g_x + cap,
      g_x + 2 * cap,
      atom.type.data(),
      cl_cell_count.data(),
      cl_cell_count_sum.data(),
      cl_cell_contents.data(),
      num_bins[0],
      num_bins[1],
      num_bins[2],
      rc_cl_inv,
      max_nb,
      d_site_NN.data(),
      d_site_NL.data());
    GPU_CHECK_KERNEL
    std::vector<int> site_NN(M);
    std::vector<int> site_NL((size_t)M * max_nb);
    d_site_NN.copy_to_host(site_NN.data(), M);
    d_site_NL.copy_to_host(site_NL.data(), (size_t)M * max_nb);

    std::vector<int> pc_slot, pc_excl, ac_slot, ac_excl;
    std::vector<int> p_off(M), p_cnt(M), a_off(M), a_cnt(M);
    std::vector<char> valid(M, 1);
    for (int m = 0; m < M; ++m) {
      int NN = site_NN[m];
      if (NN > max_nb) {
        valid[m] = 0;
        continue;
      }
      if (overlap_cutoff_active) {
        int t_ins = gcmc_types[trials[m].s_idx];
        if (t_ins < (int)overlap_cutoff2_by_type.size() && overlap_cutoff2_by_type[t_ins] > 0.0f) {
          const int s = trials[m].site;
          bool overlap = false;
          for (int i = 0; i < NN; ++i) {
            int nb = site_NL[(size_t)m * max_nb + i];
            if (nb == s) {
              continue;
            }
            double d2 = min_image_d2(
              hpos[s],
              hpos[cap + s],
              hpos[2 * cap + s],
              hpos[nb],
              hpos[cap + nb],
              hpos[2 * cap + nb]);
            if (d2 < (double)overlap_cutoff2_by_type[t_ins]) {
              overlap = true;
              break;
            }
          }
          if (overlap) {
            valid[m] = 0;
            ++num_overlap[trials[m].s_idx];
            continue;
          }
        }
      }
      int site = trials[m].site;
      p_off[m] = (int)pc_slot.size();
      a_off[m] = (int)ac_slot.size();
      for (int i = 0; i < NN; ++i) {
        int nb = site_NL[(size_t)m * max_nb + i];
        pc_slot.push_back(nb);
        pc_excl.push_back(-1);
        ac_slot.push_back(nb);
        ac_excl.push_back(site);
      }
      pc_slot.push_back(site);
      pc_excl.push_back(-1);
      p_cnt[m] = NN + 1;
      a_cnt[m] = NN;
    }

    std::vector<float> pe_present, pe_absent;
    eval_centers(pc_slot, pc_excl, pe_present);
    eval_centers(ac_slot, ac_excl, pe_absent);

    for (int m = 0; m < M; ++m) {
      int s_idx = trials[m].s_idx;
      int g = trials[m].site;
      if (!valid[m]) {
        if (trials[m].kind == INSERT) {
          int ghost = GHOST_TYPE;
          atom.type.copy_from_host(&ghost, 1, g);
          htype[g] = GHOST_TYPE;
        }
        continue;
      }
      double E_present = 0.0, E_absent = 0.0;
      for (int i = 0; i < p_cnt[m]; ++i) {
        E_present += pe_present[p_off[m] + i];
      }
      for (int i = 0; i < a_cnt[m]; ++i) {
        E_absent += pe_absent[a_off[m] + i];
      }
      double mu = gcmc_mu[s_idx];
      double e0 = gcmc_e0[s_idx];
      const double e_ref = nep_energy.energy_reference[gcmc_types[s_idx]];
      double lambda = LAMBDA_PREFACTOR / sqrt(gcmc_mass[s_idx] * temperature);
      double lambda3 = lambda * lambda * lambda;
      int N_s = trials[m].N_s;

      if (trials[m].kind == INSERT) {
        double dU = E_present - E_absent + e_ref;
        double prefactor = ratio_ins * V_cell / (lambda3 * (N_s + 1));
        double boltzmann = exp(beta * (mu - (dU - e0)));
        double acc = prefactor * boltzmann;
        bool accepted = (uni(rng) < acc);
        if (accepted) {
          ++num_ins_accept[s_idx];
          double mass = gcmc_mass[s_idx];
          double sigma = sqrt(K_B * temperature / mass);
          std::normal_distribution<double> maxwell(0.0, sigma);
          double vx = maxwell(rng), vy = maxwell(rng), vz = maxwell(rng);
          atom.mass.copy_from_host(&mass, 1, g);
          atom.velocity_per_atom.copy_from_host(&vx, 1, g);
          atom.velocity_per_atom.copy_from_host(&vy, 1, cap + g);
          atom.velocity_per_atom.copy_from_host(&vz, 1, 2 * cap + g);
          atom.cpu_type[g] = gcmc_types[s_idx];
          atom.cpu_mass[g] = mass;
          atom.cpu_atom_symbol[g] = gcmc_species[s_idx];
        } else {
          int ghost = GHOST_TYPE;
          atom.type.copy_from_host(&ghost, 1, g);
          htype[g] = GHOST_TYPE;
        }
      } else {
        int j = trials[m].site;
        double dU = E_absent - E_present - e_ref;
        double prefactor = ratio_del * lambda3 * (double)N_s / V_cell;
        double boltzmann = exp(-beta * (mu + (dU + e0)));
        double acc = prefactor * boltzmann;
        bool accepted = (uni(rng) < acc);
        if (accepted) {
          ++num_del_accept[s_idx];
          int ghost = GHOST_TYPE;
          atom.type.copy_from_host(&ghost, 1, j);
          atom.cpu_type[j] = GHOST_TYPE;
          atom.cpu_atom_symbol[j] = std::string("X");
          htype[j] = GHOST_TYPE;
          double zero = 0.0;
          atom.velocity_per_atom.copy_from_host(&zero, 1, j);
          atom.velocity_per_atom.copy_from_host(&zero, 1, cap + j);
          atom.velocity_per_atom.copy_from_host(&zero, 1, 2 * cap + j);
        }
      }
    }
  }

  write_statistics(
    md_step, htype, num_ins_attempt, num_ins_accept, num_del_attempt, num_del_accept, num_overlap);
}
