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

#include <fstream>
#include <functional>

#include <flashlight/flashlight.h>
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

#include "inference/common/IOBuffer.h"
#include "inference/decoder/Decoder.h"
#include "inference/examples/Util.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

DEFINE_string(
    input_files_base_path,
    ".",
    "path is added as prefix to input files unless the input file"
    " is a full path.");
DEFINE_string(
    feature_module_file,
    "feature_extractor.bin",
    "serialized feature extraction module.");
DEFINE_string(
    acoustic_module_file,
    "acoustic_model.bin",
    "binary file containing acoustic module parameters.");
DEFINE_string(
    transitions_file,
    "",
    "binary file containing ASG criterion transition parameters.");
DEFINE_string(tokens_file, "tokens.txt", "text file containing tokens.");
DEFINE_string(lexicon_file, "lexicon.txt", "text file containing lexicon.");
DEFINE_string(
    input_audio_file,
    "",
    "16KHz wav audio input file to be traslated to words. "
    "If no file is specified then it is read of standard input.");
DEFINE_string(silence_token, "_", "the token to use to denote silence");
DEFINE_string(
    language_model_file,
    "language_model.bin",
    "binary file containing language module parameters.");
DEFINE_string(
    decoder_options_file,
    "decoder_options.json",
    "JSON file containing decoder options"
    " including: max overall beam size, max beam for token selection, beam score threshold"
    ", language model weight, word insertion score, unknown word insertion score"
    ", silence insertion score, and use logadd when merging decoder nodes");
DEFINE_int32(
    chunk_size,
    500,
    "audio chunk size");
    
using namespace w2l;

namespace {


Mfsc &getMfsc() {
    static Mfsc mfsc(defineSpeechFeatureParams());
    return mfsc;
}

void printChunckTranscription(
    std::ostream& output,
    const std::vector<streaming::WordUnit>& wordUnits,
    int chunckStartTime,
    int chunckEndTime) {
  output << chunckStartTime << "," << chunckEndTime << ",";
  for (const auto& wordUnit : wordUnits) {
    output << wordUnit.word << " ";
  }
  output << std::endl;
}

std::string GetInputFileFullPath(const std::string& fileName) {
  return streaming::GetFullPath(fileName, FLAGS_input_files_base_path);
}

void audioStreamToWordsStream(
    std::istream& inputAudioStream,
    std::ostream& outputWordsStream,
    std::shared_ptr<streaming::Sequential> featureModule,
    std::shared_ptr<fl::Module> network,
    streaming::Decoder decoder) {
  constexpr const int lookBack = 0;
  constexpr const size_t kWavHeaderNumBytes = 44;
  constexpr const float kMaxUint16 = static_cast<float>(0x8000);
  constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
  auto kChunkSizeMsec = FLAGS_chunk_size;

  inputAudioStream.ignore(kWavHeaderNumBytes);

  const int minChunkSize = kChunkSizeMsec * kAudioWavSamplingFrequency / 1000;
  auto input = std::make_shared<streaming::ModuleProcessingState>(1);
  auto inputBuffer = input->buffer(0);
  int audioSampleCount = 0;

  decoder.start();
  bool finish = false;

  outputWordsStream << "#start (msec), end(msec), transcription" << std::endl;
  while (!finish) {
    int curChunkSize = streaming::readTransformStreamIntoBuffer<int16_t, float>(
        inputAudioStream, inputBuffer, minChunkSize, [](int16_t i) -> float {
          return static_cast<float>(i) / kMaxUint16;
        });

    if (curChunkSize < minChunkSize) {
      finish = true;
    }

    int64_t T = curChunkSize / FLAGS_channels;
    std::vector<float> values(inputBuffer->data<float>(), inputBuffer->data<float>() + T);

    auto inFeat = transpose2d<float>(values, T, FLAGS_channels, 1);

    int64_t featSz = 1;
    auto &mfsc = getMfsc();
    featSz = mfsc.getFeatureParams().mfscFeatSz();
    inFeat = mfsc.batchApply(inFeat, FLAGS_channels);
    T = inFeat.size() / (FLAGS_channels * featSz);
    if (T > 0) {
      inFeat = transpose2d<float>(inFeat, T, featSz, FLAGS_channels);
      auto inputDims = af::dim4(T, featSz, FLAGS_channels, 1);
      auto input = localNormalize(inFeat, FLAGS_localnrmlleftctx,
                                  FLAGS_localnrmlrightctx, T, 1);
      auto inputArray = af::array(inputDims, input.data());
      auto outputArr = afToVector<float>(network->forward({fl::Variable(inputArray, false)}).front());
      decoder.run(outputArr.data(), outputArr.size());
    }

    if (finish) {
      decoder.finish();
    }

    /* Print results */
    const int chunk_start_ms =
        (audioSampleCount / (kAudioWavSamplingFrequency / 1000));
    const int chunk_end_ms =
        ((audioSampleCount + curChunkSize) /
         (kAudioWavSamplingFrequency / 1000));
    printChunckTranscription(
        outputWordsStream,
        decoder.getBestHypothesisInWords(lookBack),
        chunk_start_ms,
        chunk_end_ms);
    audioSampleCount += curChunkSize;
    inputBuffer->consume<float>(curChunkSize);

    decoder.prune(lookBack);
  }
}
} // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified " + dictPath);
  }
  Dictionary tokenDict(dictPath);
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "This script currently support only CTC/ASG criterion";
  }
  int numTokens = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numTokens;

  int nFeat = 0;
  if (FLAGS_mfsc) {
    nFeat = FLAGS_filterbanks;
  } else {
    LOG(FATAL) << "This script currently support only mfsc features";
  }

  auto featureModule = std::make_shared<streaming::Sequential>();
  featureModule->add(std::make_shared<streaming::LogMelFeature>(nFeat));
  LOG_IF(FATAL, FLAGS_localnrmlleftctx <= 0)
      << "Local Norm should be used for online inference";
  featureModule->add(std::make_shared<streaming::LocalNorm>(
      nFeat, FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx));

  DecoderOptions decoderOptions;
  {
    std::ifstream decoderOptionsFile(
        GetInputFileFullPath(FLAGS_decoder_options_file));
    if (!decoderOptionsFile.is_open()) {
      throw std::runtime_error(
          "failed to open decoder options file=" +
          GetInputFileFullPath(FLAGS_decoder_options_file) + " for reading");
    }
    cereal::JSONInputArchive ar(decoderOptionsFile);
    // TODO: factor out proper serialization functionality or Cereal
    // specialization.
    ar(cereal::make_nvp("beamSize", decoderOptions.beamSize),
       cereal::make_nvp("beamSizeToken", decoderOptions.beamSizeToken),
       cereal::make_nvp("beamThreshold", decoderOptions.beamThreshold),
       cereal::make_nvp("lmWeight", decoderOptions.lmWeight),
       cereal::make_nvp("wordScore", decoderOptions.wordScore),
       cereal::make_nvp("unkScore", decoderOptions.unkScore),
       cereal::make_nvp("silScore", decoderOptions.silScore),
       cereal::make_nvp("eosScore", decoderOptions.eosScore),
       cereal::make_nvp("logAdd", decoderOptions.logAdd),
       cereal::make_nvp("criterionType", decoderOptions.criterionType));
  }

  std::vector<float> transitions;

  std::shared_ptr<const streaming::DecoderFactory> decoderFactory;
  // Create Decoder
  {
    decoderFactory = std::make_shared<streaming::DecoderFactory>(
        GetInputFileFullPath(FLAGS_tokens_file),
        "",
        GetInputFileFullPath(FLAGS_language_model_file),
        transitions,
        SmearingMode::MAX,
        FLAGS_silence_token,
        0);
  }
  auto decoder = decoderFactory->createDecoder(decoderOptions);
  audioStreamToWordsStream(
      std::cin, std::cout, featureModule, network, decoder);

  LOG(INFO) << "Done !";
  return 0;
}
