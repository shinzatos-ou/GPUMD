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
The driver class for the various MC ensembles.

Major contributors:
  Zheyong Fan: Initial implementation.
  Paul Erhart: Corrected species counts when restarting Monte Carlo.
  Shinzato: Added the grand canonical ensemble.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "mc.cuh"
#include "mc_ensemble_canonical.cuh"
#include "mc_ensemble_gcmc.cuh"
#include "mc_ensemble_sgc.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

MC::MC(void) { action_name = "mc"; }

MC::MC(
  const std::vector<std::string>& tokens,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force,
  const std::string& potential_file_name)
{
  action_name = "mc";
  if (tokens.size() > 1 && tokens[1] == "gcmc") {
    parse_mc_gcmc(tokens, group, atom, box, force, potential_file_name);
  } else {
    parse_mc(tokens, group, atom, potential_file_name);
  }
}

void MC::initialize(void)
{
  // todo
}

void MC::finalize(void) { do_mcmd = false; }

void MC::compute(int step, int num_steps, Atom& atom, Box& box, std::vector<Group>& group)
{
  if (do_mcmd) {
    if ((step + 1) % num_steps_md == 0) {
      double temperature =
        temperature_initial + step * (temperature_final - temperature_initial) / num_steps;
      mc_ensemble->compute(step + 1, temperature, atom, box, group, grouping_method, group_id);
    }
  }
}

void MC::pre_run(
  const int number_of_steps, const double, Integrate&, std::vector<Group>&, Atom&, Box&, Force&)
{
  number_of_steps_run = number_of_steps;
}

void MC::pre_force(
  const int step, const double, Integrate&, std::vector<Group>& group, Atom& atom, Box& box, Force&)
{
  compute(step, number_of_steps_run, atom, box, group);
}

void MC::parse_group(
  const std::vector<std::string>& tokens, std::vector<Group>& groups, int num_param_before_group)
{
  if (tokens[num_param_before_group] != "group") {
    PRINT_INPUT_ERROR("invalid option for mc.\n");
  }
  if (!is_valid_int(tokens[num_param_before_group + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of MCMD should be an integer.\n");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("grouping method of MCMD should >= 0.\n");
  }
  if (grouping_method >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
  }
  if (!is_valid_int(tokens[num_param_before_group + 2], &group_id)) {
    PRINT_INPUT_ERROR("group ID of MCMD should be an integer.\n");
  }
  if (group_id < 0) {
    PRINT_INPUT_ERROR("group ID of MCMD should >= 0.\n");
  }
  if (group_id >= groups[grouping_method].number) {
    PRINT_INPUT_ERROR("Group ID should < number of groups.");
  }
}

void MC::check_species_canonical(std::vector<Group>& groups, Atom& atom)
{
  bool has_multi_types = false;
  int type0 = 0;

  if (grouping_method < 0) {
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (n == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  } else {
    for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
      int n =
        groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
      if (k == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  }

  if (!has_multi_types) {
    PRINT_INPUT_ERROR("Must have more than one atom type for canonical MCMD.");
  }
}

static std::vector<std::string> get_atom_symbols_in_nep(const std::string& potential_file_name)
{
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR("MCMD only supports NEP models.");
  }

  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  std::vector<std::string> atom_symbols_in_nep(num_types);
  for (int n = 0; n < num_types; ++n) {
    atom_symbols_in_nep[n] = tokens[2 + n];
  }
  input_potential.close();
  return atom_symbols_in_nep;
}

void MC::check_species_sgc(
  std::vector<Group>& groups, Atom& atom, const std::string& potential_file_name)
{
  auto atom_symbols_in_nep = get_atom_symbols_in_nep(potential_file_name);
  for (int s = 0; s < species.size(); ++s) {
    bool allowed_species = false;
    for (int n = 0; n < atom_symbols_in_nep.size(); ++n) {
      if (species[s] == atom_symbols_in_nep[n]) {
        allowed_species = true;
        types[s] = n;
        break;
      }
    }
    if (!allowed_species) {
      PRINT_INPUT_ERROR("There are listed species not allowed in the NEP model.");
    }
  }

  if (grouping_method < 0) {
    for (int s = 0; s < species.size(); ++s) {
      for (int n = 0; n < atom.number_of_atoms; ++n) {
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  } else {
    for (int s = 0; s < species.size(); ++s) {
      for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
        int n =
          groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  }

  printf("    the initial system or specified group has:\n");
  bool has_at_least_one_species = false;
  for (int s = 0; s < species.size(); ++s) {
    printf("        %d %s atoms.\n", num_atoms_species[s], species[s].c_str());
    if (num_atoms_species[s] > 0) {
      has_at_least_one_species = true;
    }
  }

  if (!has_at_least_one_species) {
    PRINT_INPUT_ERROR(
      "Must have at least one listed species in the initial model system or specified group.");
  }
}

void MC::parse_mc(
  const std::vector<std::string>& tokens,
  std::vector<Group>& groups,
  Atom& atom,
  const std::string& potential_file_name)
{
  const int num_param = tokens.size();
  if (num_param < 6) {
    PRINT_INPUT_ERROR("mc should have at least 5 parameters.\n");
  }

  int mc_ensemble_type = 0;
  if (tokens[1] == "canonical") {
    printf("Perform canonical MCMD:\n");
    mc_ensemble_type = 0;
  } else if (tokens[1] == "sgc") {
    printf("Perform SGC MCMD:\n");
    mc_ensemble_type = 1;
  } else if (tokens[1] == "vcsgc") {
    printf("Perform VCSGC MCMD:\n");
    mc_ensemble_type = 2;
  } else {
    PRINT_INPUT_ERROR("invalid MC ensemble for MCMD.\n");
  }

  if (!is_valid_int(tokens[2], &num_steps_md)) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be an integer.\n");
  }
  if (num_steps_md <= 0) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be positive.\n");
  }

  if (!is_valid_int(tokens[3], &num_steps_mc)) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be an integer.\n");
  }
  if (num_steps_mc <= 0) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be positive.\n");
  }

  printf("    after every %d MD steps, do %d MC trials.\n", num_steps_md, num_steps_mc);

  if (!is_valid_real(tokens[4], &temperature_initial)) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be a number.\n");
  }
  if (temperature_initial <= 0) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be positive.\n");
  }

  if (!is_valid_real(tokens[5], &temperature_final)) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be a number.\n");
  }
  if (temperature_final <= 0) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be positive.\n");
  }

  printf(
    "    with an initial temperature of %g K and a final temperature of %g K.\n",
    temperature_initial,
    temperature_final);

  if (mc_ensemble_type == 1 || mc_ensemble_type == 2) {
    if (num_param < 7) {
      PRINT_INPUT_ERROR("reading error for num_types in SGC/VCSGC MCMD.\n");
    }
    if (!is_valid_int(tokens[6], &num_types_mc)) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be an integer.\n");
    }
    if (num_types_mc < 2 || num_types_mc > 4) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be 2 to 4.\n");
    }
    printf("    number of species involved in SGC/VCSGC = %d.\n", num_types_mc);

    if (num_param < (7 + num_types_mc * 2)) {
      PRINT_INPUT_ERROR("not enough (species, mu) or (species, phi) inputs.\n");
    }

    species.resize(num_types_mc);
    mu_or_phi.resize(num_types_mc);
    for (int n = 0; n < num_types_mc; ++n) {
      species[n] = tokens[7 + n * 2];
      if (!is_valid_real(tokens[7 + n * 2 + 1], &mu_or_phi[n])) {
        PRINT_INPUT_ERROR("mu or phi should be a number.\n");
      }
      printf("        species = %s, mu/phi= %g\n", species[n].c_str(), mu_or_phi[n]);
    }
  }

  if (mc_ensemble_type == 2) {
    if (num_param < 7 + num_types_mc * 2 + 1) {
      PRINT_INPUT_ERROR("Should have kappa for VCSGC.\n");
    }
    if (!is_valid_real(tokens[7 + num_types_mc * 2], &kappa)) {
      PRINT_INPUT_ERROR("kappa should be a number.\n");
    }
    if (kappa < 0) {
      PRINT_INPUT_ERROR("kappa should be positive.\n");
    }
    printf("    kappa = %g\n", kappa);
  }

  int num_param_before_group = 6;
  if (mc_ensemble_type == 1) {
    num_param_before_group = 7 + num_types_mc * 2;
  } else if (mc_ensemble_type == 2) {
    num_param_before_group = 8 + num_types_mc * 2;
  }

  if (num_param > num_param_before_group) {
    if (num_param != num_param_before_group + 3) {
      PRINT_INPUT_ERROR("reading error grouping method.\n");
    }
    parse_group(tokens, groups, num_param_before_group);
    printf("    only for atoms in group %d of grouping method %d.\n", group_id, grouping_method);
  }

  types.resize(num_types_mc);
  num_atoms_species.resize(num_types_mc, 0);
  std::fill(num_atoms_species.begin(), num_atoms_species.end(), 0);
  if (mc_ensemble_type == 0) {
    check_species_canonical(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_Canonical(tokens, num_steps_mc, potential_file_name));
  } else if (mc_ensemble_type == 1) {
    check_species_sgc(groups, atom, potential_file_name);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      tokens,
      num_steps_mc,
      false,
      species,
      types,
      num_atoms_species,
      mu_or_phi,
      kappa,
      potential_file_name));
  } else if (mc_ensemble_type == 2) {
    check_species_sgc(groups, atom, potential_file_name);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      tokens,
      num_steps_mc,
      true,
      species,
      types,
      num_atoms_species,
      mu_or_phi,
      kappa,
      potential_file_name));
  }

  do_mcmd = true;
}

static int
find_nep_type(const std::vector<std::string>& atom_symbols_in_nep, const std::string& symbol)
{
  for (int n = 0; n < (int)atom_symbols_in_nep.size(); ++n) {
    if (symbol == atom_symbols_in_nep[n]) {
      return n;
    }
  }
  return -1;
}

void MC::parse_mc_gcmc(
  const std::vector<std::string>& tokens,
  std::vector<Group>& groups,
  Atom& atom,
  Box& box,
  Force& force,
  const std::string& potential_file_name)
{
  printf("Perform GCMC MCMD:\n");

  const int num_param = tokens.size();
  if (num_param < 14) {
    PRINT_INPUT_ERROR(
      "mc gcmc needs at least: N_md T_i T_f N_sweeps a_plus n_sep max_ghost N_species "
      "(species mu mass e0)...\n");
  }

  int idx = 2;
  if (!is_valid_int(tokens[idx++], &num_steps_md) || num_steps_md <= 0) {
    PRINT_INPUT_ERROR("number of MD steps for GCMC should be a positive integer.\n");
  }
  if (!is_valid_real(tokens[idx++], &temperature_initial) || temperature_initial <= 0) {
    PRINT_INPUT_ERROR("initial temperature for GCMC should be a positive number.\n");
  }
  if (!is_valid_real(tokens[idx++], &temperature_final) || temperature_final <= 0) {
    PRINT_INPUT_ERROR("final temperature for GCMC should be a positive number.\n");
  }
  num_steps_mc = 0;
  printf(
    "    %d MD steps between GCMC calls, T from %g to %g K.\n",
    num_steps_md,
    temperature_initial,
    temperature_final);

  auto atom_symbols_in_nep = get_atom_symbols_in_nep(potential_file_name);

  int num_gcmc_sweeps = 0;
  int n_sep = 3;
  int max_ghost = 0;
  int num_species = 0;
  double a_plus = 0.5;
  if (!is_valid_int(tokens[idx++], &num_gcmc_sweeps) || num_gcmc_sweeps < 1) {
    PRINT_INPUT_ERROR("gcmc number of sweeps should be a positive integer.\n");
  }
  if (!is_valid_real(tokens[idx++], &a_plus) || a_plus <= 0 || a_plus >= 1) {
    PRINT_INPUT_ERROR("gcmc insertion probability a_plus should be in (0, 1).\n");
  }
  if (!is_valid_int(tokens[idx++], &n_sep) || n_sep < 1) {
    PRINT_INPUT_ERROR("gcmc domain separation factor should be a positive integer.\n");
  }
  if (!is_valid_int(tokens[idx++], &max_ghost) || max_ghost < 1) {
    PRINT_INPUT_ERROR("gcmc max_ghost should be a positive integer.\n");
  }
  if (!is_valid_int(tokens[idx++], &num_species) || num_species < 1) {
    PRINT_INPUT_ERROR("gcmc number of species should be a positive integer.\n");
  }
  if (idx + 4 * num_species > num_param) {
    PRINT_INPUT_ERROR("not enough (species mu mass e0) quadruples for gcmc.\n");
  }

  std::vector<std::string> gcmc_species(num_species);
  std::vector<int> gcmc_types(num_species, -1);
  std::vector<double> gcmc_mu(num_species), gcmc_mass(num_species), gcmc_e0(num_species, 0.0);
  std::vector<int> gcmc_e0_auto(num_species, 0);
  for (int s = 0; s < num_species; ++s) {
    gcmc_species[s] = tokens[idx++];
    if (!is_valid_real(tokens[idx++], &gcmc_mu[s])) {
      PRINT_INPUT_ERROR("gcmc chemical potential should be a number.\n");
    }
    if (!is_valid_real(tokens[idx++], &gcmc_mass[s]) || gcmc_mass[s] <= 0) {
      PRINT_INPUT_ERROR("gcmc mass should be a positive number.\n");
    }
    if (tokens[idx] == "auto") {
      gcmc_e0_auto[s] = 1;
      ++idx;
    } else if (!is_valid_real(tokens[idx++], &gcmc_e0[s])) {
      PRINT_INPUT_ERROR("gcmc isolated-atom energy e0 should be a number or 'auto'.\n");
    }
    gcmc_types[s] = find_nep_type(atom_symbols_in_nep, gcmc_species[s]);
    if (gcmc_types[s] < 0) {
      PRINT_INPUT_ERROR("a gcmc species is not allowed in the NEP model.\n");
    }
    const double lambda = 17.45842 / sqrt(gcmc_mass[s] * temperature_initial);
    if (gcmc_e0_auto[s]) {
      printf(
        "        species %s: mu = %g eV, mass = %g, e0 = auto, Lambda = %g A at %g K.\n",
        gcmc_species[s].c_str(),
        gcmc_mu[s],
        gcmc_mass[s],
        lambda,
        temperature_initial);
    } else {
      printf(
        "        species %s: mu = %g eV, mass = %g, e0 = %g eV, Lambda = %g A at %g K.\n",
        gcmc_species[s].c_str(),
        gcmc_mu[s],
        gcmc_mass[s],
        gcmc_e0[s],
        lambda,
        temperature_initial);
    }
  }
  printf(
    "    %d sweeps per call, a_plus = %g, n_sep = %d, max_ghost = %d.\n",
    num_gcmc_sweeps,
    a_plus,
    n_sep,
    max_ghost);

  bool gcmc_parallel = false;
  bool has_seed = false;
  int seed = 0;
  std::vector<float> overlap_cutoff2_by_type;
  while (idx < num_param) {
    if (tokens[idx] == "parallel") {
      gcmc_parallel = true;
      ++idx;
      printf("    use the GPU-parallel path.\n");
    } else if (tokens[idx] == "seed") {
      ++idx;
      if (idx >= num_param || !is_valid_int(tokens[idx++], &seed) || seed < 0) {
        PRINT_INPUT_ERROR("mc gcmc 'seed' needs a non-negative integer.\n");
      }
      has_seed = true;
      printf("    random seed = %d.\n", seed);
    } else if (tokens[idx] == "overlap_cutoff") {
      ++idx;
      int num_cutoffs = 0;
      if (idx >= num_param || !is_valid_int(tokens[idx++], &num_cutoffs) || num_cutoffs < 1) {
        PRINT_INPUT_ERROR("overlap_cutoff needs: N (species distance)...\n");
      }
      if (idx + 2 * num_cutoffs > num_param) {
        PRINT_INPUT_ERROR("not enough (species distance) pairs for overlap_cutoff.\n");
      }
      overlap_cutoff2_by_type.assign(atom_symbols_in_nep.size(), 0.0f);
      for (int k = 0; k < num_cutoffs; ++k) {
        std::string symbol = tokens[idx++];
        double d = 0.0;
        if (!is_valid_real(tokens[idx++], &d) || d < 0.0) {
          PRINT_INPUT_ERROR("overlap_cutoff distance should be a non-negative number.\n");
        }
        int t = find_nep_type(atom_symbols_in_nep, symbol);
        if (t < 0) {
          PRINT_INPUT_ERROR("an overlap_cutoff species is not allowed in the NEP model.\n");
        }
        overlap_cutoff2_by_type[t] = (float)(d * d);
        printf("    overlap_cutoff: species %s -> %g A.\n", symbol.c_str(), d);
      }
    } else {
      PRINT_INPUT_ERROR(
        "unknown option for mc gcmc (expected parallel, seed or overlap_cutoff).\n");
    }
  }

  if (!MC_Ensemble_GCMC::capacity_reserved() && force.get_number_of_potentials() > 0) {
    PRINT_INPUT_ERROR("mc gcmc must appear before potential in the first run.\n");
  }

  MC_Ensemble_GCMC* gcmc = new MC_Ensemble_GCMC(tokens, num_steps_mc, potential_file_name);
  if (has_seed) {
    gcmc->set_seed((unsigned)seed);
  }
  if (!overlap_cutoff2_by_type.empty()) {
    gcmc->set_overlap_cutoffs(overlap_cutoff2_by_type);
  }
  gcmc->set_gcmc_parameters(
    gcmc_species,
    gcmc_types,
    gcmc_mu,
    gcmc_mass,
    gcmc_e0,
    gcmc_e0_auto,
    num_gcmc_sweeps,
    a_plus,
    n_sep,
    max_ghost,
    gcmc_parallel);
  gcmc->ensure_gcmc_capacity(atom, box, groups);

  mc_ensemble.reset(gcmc);
  do_mcmd = true;
}
