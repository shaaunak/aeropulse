#!/usr/bin/env bash
set -euo pipefail
# Official NASA portal contains CMAPSSData.zip; direct archive URL may change.
URL='https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
DEST="${1:-data/raw}"
mkdir -p "$DEST"
printf 'Downloading publicly hosted NASA C-MAPSS archive into %s\n' "$DEST"
curl --fail --location --retry 2 "$URL" -o "$DEST/cmapss_outer.zip"
unzip -o "$DEST/cmapss_outer.zip" -d "$DEST/extracted"
find "$DEST/extracted" -type f \( -iname '*CMAPSS*.zip' -o -iname '*Damage*.zip' \) -print | while read -r file; do
  echo "Extracting nested ZIP $file"
  unzip -o "$file" -d "$DEST/extracted/inner"
done
find "$DEST" -type f \( -name 'train_FD001.txt' -o -name 'test_FD001.txt' -o -name 'RUL_FD001.txt' \) -print
printf 'Confirm source, terms and checksums before use; do not commit raw data to git.\n'
