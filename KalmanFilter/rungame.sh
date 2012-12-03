choice=$1

choice="sitting"

make --directory=src/


if [ "$choice"  = "sitting" ] ; then 

	../../bzrflag/bin/bzrflag --world=../../bzrflag/maps/empty.bzw --friendly-fire --red-port=50100 --red-tanks=1 --green-port=50101 --green-tanks=0 --purple-port=50102 --purple-tanks=1 --blue-port=50103 --blue-tanks=0 $@ &
	
	sleep 2


elif [ "$choice" = "straight" ] ; then

	../../bzrflag/bin/bzrflag --world=../../bzrflag/maps/empty.bzw --default-posnoise=5 --default-true-positive=.97 --default-true-negative=.9 --occgrid-width=100 --no-report-obstacles --friendly-fire --red-port=50100 --red-tanks=1 --green-port=50101 --green-tanks=1 --purple-port=50102 --purple-tanks=0 --blue-port=50103 --blue-tanks=0 $@ &

	sleep 2
											
	python src/SimpleAgent.py localhost 50101 &

else
	
	../../bzrflag/bin/bzrflag --world=../../bzrflag/maps/empty.bzw --default-posnoise=5 --default-true-positive=.97 --default-true-negative=.9 --occgrid-width=100 --no-report-obstacles --friendly-fire --red-port=50100 --red-tanks=1 --green-port=50101 --green-tanks=1 --purple-port=50102 --purple-tanks=0 --blue-port=50103 --blue-tanks=0 $@ &
			
	sleep 2
						
	python src/WildPigeonAgent.py localhost 50101 & # change this with non-conforming agent

fi


python src/kalman_filter_agent.py localhost 50100 &
				