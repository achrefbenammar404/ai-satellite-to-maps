#!/usr/bin/env zsh

# Variables
url="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
save_path="data/maps.tar.gz"        # Adjusting path to a local 'data' directory
extract_path="data/maps_dataset"

# Create the directories if they don't exist
mkdir -p "$(dirname "$save_path")"
mkdir -p "$extract_path"

# Download the file using curl
echo "Downloading the dataset from $url..."
curl -L "$url" -o "$save_path"

# Extract the tar.gz file
echo "Extracting the dataset to $extract_path..."
tar -xzf "$save_path" -C "$extract_path"

echo "Download and extraction complete!"
