../../bzrflag/bin/bzrflag --world=../../bzrflag/maps/empty.bzw --default-true-positive=.97 --default-true-negative=.9 --occgrid-width=100 --no-report-obstacles --friendly-fire --red-port=50100 --red-tanks=3 --green-port=50101 --green-tanks=0 --purple-port=50102 --purple-tanks=0 --blue-port=50103 --blue-tanks=0 $@ &

sleep 2

make --directory=src/

python src/grid_filter_agent.py localhost 50100 &
#compiled/grid_filter_agent.pyc localhost 50100 &
