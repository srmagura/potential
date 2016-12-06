# Don't actually run this file... the alt extension will not be used
# This is here for convenience; copy and paste the commands into the terminal
python3 runz.py ihz-bessel-line inner-sine -c 2048 > ../output/ihz-bessel-line_inner-sine.txt &
python3 runz.py ihz-bessel-line inner-sine -c 2048 > ../output/ihz-bessel-line_inner-sine_alt.txt &
python3 runz.py ihz-bessel-line cubic -c 2048 > ../output/ihz-bessel-line_cubic.txt &

python3 runz.py ihz-bessel-quadratic outer-sine -c 2048 > ../output/ihz-bessel-quadratic_outer-sine.txt &
python3 runz.py ihz-bessel-quadratic sine7 -c 2048 > ../output/ihz-bessel-quadratic_sine7.txt &
