#!/usr/bin/env bash
# Download the weights exps 15/16 need into the HF cache layout the code expects (/weights/hf), on a
# bare GPU machine (RunPod etc.). ~150 GB; needs a Hugging Face token in HF_TOKEN for rate limits.
#   export HF_TOKEN=hf_...; bash infra/download_weights_local.sh
set -euo pipefail
export HF_HOME="${HF_HOME:-/weights/hf}"
export HF_HUB_OFFLINE=0
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$HF_HOME"
DL="hf download"; command -v hf >/dev/null 2>&1 || DL="huggingface-cli download"
$DL Qwen/Qwen3.6-27B                                                                     # ~55.6 GB
$DL ceselder/qwen3.6-27b-nla-rl --include "av_base/*" "av_rl_adapters/iter_000400/*" \
    "ar_reconstructor/*" "nla_meta.yaml" "run_config.yaml" "README.md"                   # ~93 GB
$DL agu18dec/olens_and_ar                                                                # ~1 GB
$DL neuronpedia/jacobian-lens --include "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
echo "done; snapshots:"; ls "$HF_HOME/hub" | grep -E "models--"
