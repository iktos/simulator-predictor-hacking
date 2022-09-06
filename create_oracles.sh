chids='dataset1 dataset2 dataset3 dataset4'
powers='6 15'
seeds=$(seq 19 23)

for chid in $chids
do
for power in $powers
do
for seed in $seeds
do
echo "Running $chid $power $seed"
python create_oracle.py $chid 19 2 $power && python create_oracle.py $chid $seed 1 $power
done
done
done