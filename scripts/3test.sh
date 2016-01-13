#!/bin/bash

MINPARAMS=1
if [ $# -lt "$MINPARAMS" ]
then
  echo "Required argument: k"
  exit 1
fi

c=4096

python3 run.py sing-h-parabola -a -o 2 -c $c -r > ../sing-h-parabola_a_o2_k$1.txt &
disown

python3 run.py sing-h-parabola -o 2 -c $c -r > ../sing-h-parabola_o2_k$1.txt &
disown

python3 run.py sing-h-hat -a -o 2 -c $c -r > ../sing-h-hat_a_o2_k$1.txt &
disown
