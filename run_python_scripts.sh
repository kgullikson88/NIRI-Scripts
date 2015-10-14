set -e

for f in N*.fits
do
    python cleanir.py $f
done

python classify.py --skip-first c*.fits

for f in `cat OBJECT*.list`
do
    python nirlin.py $f
done

python classify.py l*.fits