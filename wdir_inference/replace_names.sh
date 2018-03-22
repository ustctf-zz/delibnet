folder_name=$1

for ff in ${folder_name}/*; do
mv $ff  ${ff/delib.1.2.9/model}
done
