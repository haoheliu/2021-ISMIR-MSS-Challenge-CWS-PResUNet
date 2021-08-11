dir=$2
type=$1
a=`ls $dir`
for step in $a
do
  if [[ $2 == */ ]]; then
    python3 evaluator/eval.py --step $(pwd)/$dir$step --type $type
  else
    python3 evaluator/eval.py  --step $(pwd)/$dir/$step --type $type
  fi
done
wait