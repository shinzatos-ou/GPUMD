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
The GCMC_Driver class splits a run.in file with mc gcmc into sub-runs of N_md
steps and performs grand canonical insertions and deletions between consecutive
sub-runs.
------------------------------------------------------------------------------*/

#pragma once
#include <memory>
#include <string>
#include <vector>

class RunInput;
class MC_Ensemble_GCMC;

class GCMC_Driver
{
public:
  GCMC_Driver(void);
  ~GCMC_Driver(void);

  int num_subruns = 0;

  bool has_gcmc(const RunInput& run_input);
  void initialize(const RunInput& run_input);
  RunInput prepare_subrun(int run_idx);
  void mc_step(int run_idx);

private:
  void parse_gcmc(const std::vector<std::string>& tokens);
  void analyze_run(const RunInput& run_input);

  std::vector<std::string> input_raw_lines;
  std::vector<std::vector<std::string>> subrun_lines;
  std::string potential_file_name;
  std::vector<std::string> atom_symbols_in_nep;
  std::unique_ptr<MC_Ensemble_GCMC> gcmc;

  int num_steps_md = 0;
  int num_steps_total = 0;
  double temperature_initial = 0.0;
  double temperature_final = 0.0;
  bool has_group = false;
};
