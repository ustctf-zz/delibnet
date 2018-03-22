for ii in *.decodes; do
  new_name=$(echo ${ii} | grep -o -P ".*?(iter[0-9]*)")
  mv ${ii} ${new_name}
done
