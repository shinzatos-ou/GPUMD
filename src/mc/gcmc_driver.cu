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
Split a run.in file with mc gcmc into sub-runs of N_md steps and perform grand
canonical insertions and deletions between consecutive sub-runs.

Major contributors:
  Shinzato: Initial implementation.
------------------------------------------------------------------------------*/

#include "gcmc_driver.cuh"
#include "mc_ensemble_gcmc.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/read_xyz.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_io/compact_nep.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>

static const char* SUBRUN_XYZ = "gcmc_subrun.xyz";

static void copy_file(const std::string& in_file, const std::string& out_file)
{
  std::ifstream input(in_file, std::ios::binary);
  if (!input.is_open()) {
    std::string error = "Failed to open " + in_file + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::ofstream output(out_file, std::ios::binary);
  if (!output.is_open()) {
    std::string error = "Failed to open " + out_file + " for writing.";
    PRINT_INPUT_ERROR(error.c_str());
  }
  output << input.rdbuf();
}

static bool model_has_group(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open model.xyz.");
  }
  std::string line;
  std::getline(input, line);
  std::getline(input, line);
  std::string lower = line;
  std::transform(
    lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
  const std::string marker = "properties=";
  const size_t start = lower.find(marker);
  if (start == std::string::npos) {
    return false;
  }
  size_t end = start + marker.size();
  while (end < lower.size() && !std::isspace(static_cast<unsigned char>(lower[end]))) {
    ++end;
  }
  std::string properties = lower.substr(start + marker.size(), end - start - marker.size());
  std::vector<std::string> fields;
  size_t pos = 0;
  while (pos <= properties.size()) {
    size_t colon = properties.find(':', pos);
    if (colon == std::string::npos) {
      colon = properties.size();
    }
    fields.emplace_back(properties.substr(pos, colon - pos));
    pos = colon + 1;
  }
  for (size_t n = 0; n + 2 < fields.size(); n += 3) {
    if (fields[n] == "group") {
      return true;
    }
  }
  return false;
}

static int find_nep_type(const std::vector<std::string>& atom_symbols, const std::string& symbol)
{
  for (int n = 0; n < (int)atom_symbols.size(); ++n) {
    if (atom_symbols[n] == symbol) {
      return n;
    }
  }
  return -1;
}

static std::string get_first_potential_file(const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.size() >= 2 && line.tokens[0] == "potential") {
      return line.tokens[1];
    }
  }
  PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  return "";
}

static bool is_one_shot_keyword(const std::string& keyword)
{
  static const char* one_shot_keywords[] = {
    "replicate",
    "change_box",
    "minimize",
    "compute_phonon",
    "compute_hessian",
    "compute_cohesive",
    "compute_elastic"};
  for (const char* one_shot_keyword : one_shot_keywords) {
    if (keyword == one_shot_keyword) {
      return true;
    }
  }
  return false;
}

static std::string format_real(const double value)
{
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

static std::vector<int> get_schedule_indices(const std::vector<std::string>& tokens)
{
  std::vector<int> indices;
  if (tokens.size() < 2) {
    return indices;
  }
  const std::string& name = tokens[1];
  if (
    name == "nvt_ber" || name == "npt_ber" || name == "nvt_nhc" || name == "nvt_lan" ||
    name == "nvt_bdp" || name == "nvt_bao" || name == "nvt_qtb" || name == "npt_scr") {
    indices.push_back(2);
  } else if (name == "pimd" || name == "pimd_scr") {
    indices.push_back(3);
  } else if (name == "nvt_mttk" || name == "npt_mttk" || name == "nph_mttk" || name == "npt_qtb") {
    int i = 2;
    while (i < (int)tokens.size()) {
      const std::string& key = tokens[i];
      if (key == "couple") {
        i += 2;
      } else if (
        key == "temp" || key == "iso" || key == "aniso" || key == "tri" || key == "x" ||
        key == "y" || key == "z" || key == "xy" || key == "xz" || key == "yz") {
        indices.push_back(i + 1);
        i += 3;
      } else {
        ++i;
      }
    }
  }
  return indices;
}

static std::string
get_subrun_ensemble_line(const RunInputLine& line, const int subrun, const int num_subruns)
{
  std::vector<std::string> tokens = line.tokens;
  bool has_schedule = false;
  for (const int i : get_schedule_indices(tokens)) {
    double start = 0.0;
    double stop = 0.0;
    if (
      i + 1 >= (int)tokens.size() || !is_valid_real(tokens[i], &start) ||
      !is_valid_real(tokens[i + 1], &stop) || start == stop) {
      continue;
    }
    has_schedule = true;
    tokens[i] = format_real(start + (stop - start) * subrun / num_subruns);
    tokens[i + 1] = format_real(start + (stop - start) * (subrun + 1) / num_subruns);
  }
  if (!has_schedule) {
    return line.raw_line;
  }
  std::string result = tokens[0];
  for (size_t n = 1; n < tokens.size(); ++n) {
    result += " " + tokens[n];
  }
  return result;
}

static void write_model(
  const std::string& filename,
  const int num_original_atoms,
  Atom& atom,
  const Box& box,
  const std::vector<Group>& groups)
{
  const int cap = atom.number_of_atoms;
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());

  int num_atoms = 0;
  for (int n = 0; n < cap; ++n) {
    if (atom.cpu_type[n] >= 0) {
      ++num_atoms;
    }
  }
  if (num_atoms < 2) {
    PRINT_INPUT_ERROR("mc gcmc left fewer than 2 atoms, which the next sub-run cannot use.\n");
  }

  FILE* fid = my_fopen(filename.c_str(), "w");
  fprintf(fid, "%d\n", num_atoms);
  const double* h = box.cpu_h;
  fprintf(
    fid,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" pbc=\"%c %c %c\" "
    "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3",
    h[0],
    h[3],
    h[6],
    h[1],
    h[4],
    h[7],
    h[2],
    h[5],
    h[8],
    box.pbc_x ? 'T' : 'F',
    box.pbc_y ? 'T' : 'F',
    box.pbc_z ? 'T' : 'F');
  if (!groups.empty()) {
    fprintf(fid, ":group:I:%d", (int)groups.size());
  }
  fprintf(fid, "\n");

  const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
  for (int n = 0; n < cap; ++n) {
    if (atom.cpu_type[n] < 0) {
      continue;
    }
    fprintf(
      fid,
      "%s %.17g %.17g %.17g %.17g %.17g %.17g %.17g",
      atom.cpu_atom_symbol[n].c_str(),
      atom.cpu_position_per_atom[n],
      atom.cpu_position_per_atom[n + cap],
      atom.cpu_position_per_atom[n + 2 * cap],
      atom.cpu_mass[n],
      atom.cpu_velocity_per_atom[n] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + cap] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + 2 * cap] * natural_to_A_per_fs);
    for (const auto& group : groups) {
      fprintf(fid, " %d", n < num_original_atoms ? group.cpu_label[n] : 0);
    }
    fprintf(fid, "\n");
  }
  fclose(fid);
}

GCMC_Driver::GCMC_Driver(void) {}

GCMC_Driver::~GCMC_Driver(void) {}

bool GCMC_Driver::has_gcmc(const RunInput& run_input)
{
  for (const auto* line : run_input.find_all("mc")) {
    if (line->tokens.size() >= 2 && line->tokens[1] == "gcmc") {
      return true;
    }
  }
  return false;
}

void GCMC_Driver::parse_gcmc(const std::vector<std::string>& tokens)
{
  printf("Perform GCMC MCMD:\n");

  const int num_param = tokens.size();
  if (num_param < 13) {
    PRINT_INPUT_ERROR("mc gcmc needs at least: N_md T_i T_f N_sweeps a_plus n_sep N_species "
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
  printf(
    "    %d MD steps between GCMC calls, T from %g to %g K.\n",
    num_steps_md,
    temperature_initial,
    temperature_final);

  int num_gcmc_sweeps = 0;
  int n_sep = 3;
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
  printf("    %d sweeps per call, a_plus = %g, n_sep = %d.\n", num_gcmc_sweeps, a_plus, n_sep);

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
  if (gcmc_parallel && n_sep < 2) {
    PRINT_INPUT_ERROR("n_sep of mc gcmc should be at least 2 with parallel.\n");
  }

  gcmc.reset(new MC_Ensemble_GCMC(tokens, 0, potential_file_name));
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
    gcmc_parallel);

  for (const auto& symbol : gcmc_species) {
    register_compact_nep_required_species(symbol);
  }
}

void GCMC_Driver::analyze_run(const RunInput& run_input)
{
  int gcmc_line = -1;
  int run_line = -1;

  const std::vector<RunInputLine>& input_lines = run_input.lines();
  for (int n = 0; n < (int)input_lines.size(); ++n) {
    const std::vector<std::string>& tokens = input_lines[n].tokens;
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] == "mc") {
      if (gcmc_line >= 0 || tokens.size() < 2 || tokens[1] != "gcmc") {
        PRINT_INPUT_ERROR("mc gcmc should be the only mc keyword in run.in.\n");
      }
      gcmc_line = n;
    } else if (tokens[0] == "deposit") {
      PRINT_INPUT_ERROR("mc gcmc cannot be used together with deposit.\n");
    } else if (tokens[0] == "dump_observer" && tokens.size() >= 2 && tokens[1] == "average") {
      PRINT_INPUT_ERROR("mc gcmc cannot be used together with dump_observer average.\n");
    } else if (tokens[0] == "run") {
      if (run_line >= 0) {
        PRINT_INPUT_ERROR("run should appear only once in run.in when mc gcmc is used.\n");
      }
      if (tokens.size() < 2 || !is_valid_int(tokens[1], &num_steps_total)) {
        PRINT_INPUT_ERROR("number of steps should be an integer.\n");
      }
      run_line = n;
    } else if (is_one_shot_keyword(tokens[0]) && run_line >= 0) {
      const std::string error = tokens[0] + " should appear before run when mc gcmc is used.\n";
      PRINT_INPUT_ERROR(error.c_str());
    }
  }
  if (run_line < 0) {
    PRINT_INPUT_ERROR("run.in should contain a run keyword when mc gcmc is used.\n");
  }

  parse_gcmc(input_lines[gcmc_line].tokens);

  if (num_steps_total < num_steps_md || num_steps_total % num_steps_md != 0) {
    PRINT_INPUT_ERROR("number of steps in run should be a multiple of N_md of mc gcmc.\n");
  }
  num_subruns = num_steps_total / num_steps_md;

  std::string dump_line =
    "dump_xyz " + std::to_string(num_steps_md) + " " + SUBRUN_XYZ + " mass velocity";
  if (has_group) {
    dump_line += " group_labels";
  }
  dump_line += " precision double";

  subrun_lines.assign(num_subruns, std::vector<std::string>());
  for (int k = 0; k < num_subruns; ++k) {
    std::vector<std::string>& lines = subrun_lines[k];
    for (int n = 0; n < (int)input_lines.size(); ++n) {
      const std::vector<std::string>& tokens = input_lines[n].tokens;
      if (n == gcmc_line) {
        continue;
      }
      if (n == run_line) {
        lines.emplace_back(dump_line);
        lines.emplace_back("run " + std::to_string(num_steps_md));
      } else if (tokens.empty()) {
        lines.emplace_back(input_lines[n].raw_line);
      } else if (tokens[0] == "ensemble") {
        lines.emplace_back(get_subrun_ensemble_line(input_lines[n], k, num_subruns));
      } else if (k == 0 || !is_one_shot_keyword(tokens[0])) {
        lines.emplace_back(input_lines[n].raw_line);
      }
    }
  }
}

void GCMC_Driver::initialize(const RunInput& run_input)
{
  if (std::ifstream("model.xyz.original").is_open()) {
    PRINT_INPUT_ERROR(
      "model.xyz.original already exists; move or remove it before running mc gcmc.\n");
  }

  for (const auto& line : run_input.lines()) {
    input_raw_lines.emplace_back(line.raw_line);
  }
  potential_file_name = get_first_potential_file(run_input);
  atom_symbols_in_nep = get_atom_symbols(potential_file_name);
  has_group = model_has_group("model.xyz");
  std::remove(SUBRUN_XYZ);

  analyze_run(run_input);
  copy_file("model.xyz", "model.xyz.original");
  printf("Split run.in into %d sub-runs of %d steps for mc gcmc.\n", num_subruns, num_steps_md);
}

RunInput GCMC_Driver::prepare_subrun(int run_idx)
{
  printf("Running sub-run %d / %d.\n", run_idx + 1, num_subruns);
  fflush(stdout);

  return RunInput(subrun_lines[run_idx]);
}

void GCMC_Driver::mc_step(int run_idx)
{
  copy_file(SUBRUN_XYZ, "model.xyz");
  std::remove(SUBRUN_XYZ);

  RunInput run_input(input_raw_lines);
  Atom atom;
  Box box;
  std::vector<Group> groups;
  GPU_Vector<double> thermo;
  int has_velocity_in_xyz = 0;
  int number_of_types = 0;
  initialize_position(run_input, has_velocity_in_xyz, number_of_types, box, groups, atom);
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    atom.cpu_type[n] = find_nep_type(atom_symbols_in_nep, atom.cpu_atom_symbol[n]);
  }
  allocate_memory_gpu(groups, atom, thermo);

  const int num_original_atoms = atom.number_of_atoms;
  gcmc->add_ghost_slots(atom, box, gcmc->get_max_insertions(box));

  const int md_step = (run_idx + 1) * num_steps_md;
  const double temperature = temperature_initial + (md_step - 1) *
                                                     (temperature_final - temperature_initial) /
                                                     num_steps_total;
  gcmc->compute(md_step, temperature, atom, box, groups, -1, -1);

  write_model("model.xyz", num_original_atoms, atom, box, groups);
}
