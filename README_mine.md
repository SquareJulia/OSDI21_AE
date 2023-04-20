Preprocessing result stored at:
```
preprocessed/{dataDir}/{graphName}/
{num_features}_{hidden}_{num_classes}_{strategy}_{density}_{tileDim}

eg. 
preprocessed/
    osdi-ae-graphs/citeseer/
        1000_16_7_rb_0.001_32/
            dataset.pt
            inputInfo.pt
            inputLayerSpRT.pt
            hiddenLayerSpRT.pt
            SparseRT/
                AB.npz
                degrees.npy
                inputLayer.ptx
                inputLayer.cubin
                hiddenLayer.ptx
                hiddenLayer.cubin
```


cd METIS_module/pymetis
./configure.py
make
sudo make install
PYTHONPATH={}/OSDI21_AE/METIS_module/pymetis/build/lib.linux-x86_64-cpython-39