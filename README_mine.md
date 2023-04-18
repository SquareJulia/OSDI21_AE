```
{graph_name}_{C_dim}_{degree?d:x}{rabbit?r:x}{rabbit?rabbitBarrier:}_{density}_{A_tileDim}
```
eg. 
citeseer_16_dr45_0.003_80.ptx
citeseer_16_dr45_0.003_80.cubin

cd METIS_module/pymetis
./configure.py
make
sudo make install
PYTHONPATH={}/OSDI21_AE/METIS_module/pymetis/build/lib.linux-x86_64-cpython-39