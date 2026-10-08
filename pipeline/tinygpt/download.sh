#!/usr/bin/env bash
# Download TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0) from Hugging Face.
# valid split in full (~22 MB) + first ~100 MB of the train split via an HTTP Range request.
# Retries with backoff because Hugging Face sometimes resets connections from this network.
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data/tinystories"
mkdir -p "$D"
BASE="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
fetch() { # url out [range]
  local url="$1" out="$2" range="${3:-}" i
  for i in 1 2 3 4 5 6; do
    if [ -n "$range" ]; then
      curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -r "$range" -o "$out.part" "$url" && mv "$out.part" "$out" && return 0
    else
      curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -C - -o "$out.part" "$url" && mv "$out.part" "$out" && return 0
    fi
    echo "attempt $i failed for $url; sleeping $((i*10))s"; sleep $((i*10))
  done
  return 1
}
[ -s "$D/TinyStoriesV2-GPT4-valid.txt" ] || fetch "$BASE/TinyStoriesV2-GPT4-valid.txt" "$D/TinyStoriesV2-GPT4-valid.txt" || echo "VALID FAILED"
[ -s "$D/train_head_100MB.raw.txt" ] || fetch "$BASE/TinyStoriesV2-GPT4-train.txt" "$D/train_head_100MB.raw.txt" "0-104857599" || echo "TRAIN FAILED"
ls -la "$D"
