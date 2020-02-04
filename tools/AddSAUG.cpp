/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

#include <flashlight/flashlight.h>
#include <flashlight/contrib/contrib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "libraries/common/Dictionary.h"
#include "module/SpecAugment.h"
#include "module/TDSBlock.h"
#include "module/W2lModule.h"
#include "runtime/runtime.h"
#include "common/FlashlightUtils.h"
#include "module/SpecAugment.h"


DEFINE_string(outdir, "", "");
DEFINE_string(archfile, "", "NEW Arch file");

using namespace w2l;

namespace {

} // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::Module> seq;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  W2lSerializer::load(FLAGS_am, cfg, network, criterion, netoptim, critoptim);
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);
  
  auto numFeatures = getSpeechFeatureSize();
  // Encoder network, works on audio
  seq = createW2lSeqModule(FLAGS_archfile, 80, 0);

  auto params = network->params();
  
  for (int i = 0; i < params.size(); ++i) {
      seq->setParams(params[i], i);
  }
  
  LOG(INFO) << "Added SAUG Layer";
  LOG(INFO) << "[Network] " << seq->prettyString();
  
  std::string amFilePath = pathsConcat(FLAGS_outdir, "acoustic_model_saug.bin");
  W2lSerializer::save(
            amFilePath, cfg, seq, criterion, netoptim, critoptim);


  LOG(INFO) << "Done !";
  return 0;
}
