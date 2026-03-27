#!/usr/bin/env bash
set -e

# Configuration
FILE_NAME="speech_commands_v0.01.tar.gz"
PRIMARY_URL="https://storage.googleapis.com/download.tensorflow.org/data/${FILE_NAME}"
FALLBACK_URL="https://github.com/Paperspace/speech-commands-files/releases/download/v0.01/${FILE_NAME}"
DATASET_FOLDER="datasets/speech_commands"
ARCHIVE_PATH="${DATASET_FOLDER}/${FILE_NAME}"
AUDIO_DIR="${DATASET_FOLDER}/audio"

# Create the target folder
mkdir -p "${DATASET_FOLDER}"

# Download if not already present
if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "→ Downloading ${FILE_NAME} from primary URL..."
  n=0
  until [ $n -ge 5 ]
  do
    if curl -fSL --retry 3 --retry-delay 5 --retry-max-time 60 \
         "${PRIMARY_URL}" -o "${ARCHIVE_PATH}"; then
      echo "✔ Downloaded from primary URL"
      break
    else
      echo "⚠ Primary download failed, retry $((n+1))..."
      n=$((n+1))
      sleep $((2**n))
    fi
  done

  if [ ! -f "${ARCHIVE_PATH}" ]; then
    echo "→ Trying fallback URL..."
    curl -fSL --retry 3 --retry-delay 5 "${FALLBACK_URL}" -o "${ARCHIVE_PATH}"
    echo "✔ Downloaded from fallback URL"
  fi
else
  echo "✔ Archive already exists at ${ARCHIVE_PATH}"
fi

# Extract
echo "→ Extracting ${FILE_NAME} to ${AUDIO_DIR}..."
mkdir -p "${AUDIO_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${AUDIO_DIR}"

# Split train/valid/test
echo "→ Splitting into train/valid/test..."
python "${DATASET_FOLDER}/split_dataset.py" "${DATASET_FOLDER}"

# Merge validation into train (move individual files)
if [ -d "${DATASET_FOLDER}/valid" ]; then
  echo "→ Merging validation into train..."
  # For each class subfolder
  for cls in "${DATASET_FOLDER}/valid/"*; do
    [ -d "${cls}" ] || continue
    class_name=$(basename "${cls}")
    mkdir -p "${DATASET_FOLDER}/train/${class_name}"
    # Move each WAV inside valid/<class> into train/<class>
    for wav in "${cls}/"*.wav; do
      mv "$wav" "${DATASET_FOLDER}/train/${class_name}/"
    done
  done
  # Clean up
  rm -rf "${DATASET_FOLDER}/valid"
fi

echo "✅ Done."