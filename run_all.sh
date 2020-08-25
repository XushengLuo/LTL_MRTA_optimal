y#!/bin/bash

#for w in 50
#do
#    echo "---------------------------w=${w}-------------------------"
#    for n in  12
#    do
#        echo "--------------------n=${n}-----------------------"
#        fc
#        do
#            /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/LTL_MRTA_plus/stap.py ${w} ${n}
#        done
#    done
#    # for ((n=3;n<4;n++))
#done
#     echo "---------------------------n=${n}-------------------------"
#     for h in 10 15 20
#     do
#         echo "---------------------h=${h}-----------------------------"
#         /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/GitHub/RRT*_LTL/SMT4MulR_2.py ${h}
#     done
# done

# ------------------------- case 1 ------------------------
#for ((n=1;n<50;n++))
#do
#    /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6    /Users/chrislaw/Github/LTL_MRTA_optimal/case1.py
#done

# -------------------------- case 2 ------------------------
for ((n=1;n<21;n++))
do
    echo "---------------------------n=${n}-------------------------"
    /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/LTL_MRTA_optimal/case2.py 'f'
done
# -------------------------- case 3 ------------------------
# for N in 32 48
# do
#     echo "---------------------------N=${N}-------------------------"
#     for ((i=1; i<6; i++))
#     do
#         echo "---------------------------i=${i}-------------------------"

#         /Applications/MATLAB_R2017b.app/bin/matlab -nodesktop -nosplash -r "cd('/Users/chrislaw/Box Sync/Research/2020_LTL_MRTA_IJRR/cLTL-hierarchical-master'); example_grid_random(${N}, 7, 35, 10);exit"
#         for m in 'f' 'p'
#         do
#             echo "---------------------------m=${m}-------------------------"
#             /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/LTL_MRTA_optimal/case3_2.py ${m} ${N}
#         done
#     done
# done
