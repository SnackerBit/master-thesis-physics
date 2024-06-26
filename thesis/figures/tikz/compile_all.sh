#!/usr/bin/env bash

shopt -s globstar

for d in **/ ; do
  (
    #echo $d;
    cd $d;
    if ls *.tex; then
        for f in *.tex; do
            pdflatex $f
        done
    fi
  )
done

#find . -name '*.tex' -exec pdflatex "{}" \;
#find . -type d -exec sh -c "(cd {}; for i in *.tex; do pdflatex "$i"; done)" \;
