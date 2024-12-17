#!/bin/bash

# check if the markdown folder exists if not create it
if [ ! -d "markdown" ]; then
  mkdir markdown
fi

# convert all the notebooks to markdown
for notebook in notebooks/IR_02_*.ipynb; do
  jupyter nbconvert --to markdown "$notebook" --output-dir markdown
done