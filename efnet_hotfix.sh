#!/bin/bash

# Gliched files
file_path="/usr/local/lib/python3.11/site-packages/efficientnet/model.py"
file_path2="/usr/local/lib/python3.11/site-packages/efficientnet/__init__.py"

if [ ! -f "$file_path" ]; then
    echo "File $file_path not found."
    exit 1
fi

sed -i 's/init_keras_custom_objects/init_tfkeras_custom_objects/g' "$file_path"

sed -i 's/return x \* backend\.sigmoid(x)/return x \* tf.keras.backend.sigmoid(x)/g' "$file_path"

sed -i '/^from __future__ import/ { :a; N; /\n\n/!ba; s/\(from __future__ import.*\)\(\n\n\)/\1\2import tensorflow as tf\n\n/ }' "$file_path"

sed -i 's/keras.utils.generic_utils.get_custom_objects()/keras.utils.get_custom_objects()/g' "$file_path2"

echo "Modifications complete."
