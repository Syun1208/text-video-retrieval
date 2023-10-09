#!/bin/bash
FILES=/home/hoangtv/Desktop/Long/text-video-retrieval/data/news_aic2023/*

for f in $FILES
do
  if [[ "$f" != *\.* ]]
  then
    unzip "$f"
  fi
done