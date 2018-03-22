all_trans=$(ls . | grep "alpha[0-9\.]*_iter[0-9]*$")

for fname in ${all_trans}; do
if [ -f ${fname}.beamouts ]; then
    continue
else
    python split_beams.py $fname
fi

done
