cd GNNConv/
sudo rm -rf build dist GNNAdvisor.egg-info
TORCH_CUDA_ARCH_LIST="7.0" python setup.py clean --all install
cd ../
