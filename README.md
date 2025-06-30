

### Line-command

run GEMI with alignment by partial reconstruction.

```unix
python run_GEMI_APR.py --proj CRC-DX  --num_epochs 15 --plr 1e-3 --elr 1e-3  --smoothing 0.05
python run_GEMI_APR.py --proj STAD-DX --num_epochs 25 --plr 1e-3 --elr 1e-3 --smoothing 0.1
python run_GEMI_APR.py --proj GBM-DX --num_epochs 50 --plr 5e-3 --elr 5e-3 --smoothing 0.01
```

run GEMI with alignment by orthogonal decomposition and partial reconstruction.

```
python run_GEMI_AODPR.py --proj CRC-DX  --num_epochs 15 --plr 1e-3 --elr 1e-3  --smoothing 0.05
python run_GEMI_AODPR.py --proj STAD-DX --num_epochs 25 --plr 1e-3 --elr 1e-3 --smoothing 0.1 
python run_GEMI_AODPR.py --proj GBM-DX --num_epochs 50 --plr 5e-3 --elr 5e-3 --smoothing 0.01
```

run GEMI with alignment by orthogonal decomposition.

```
python run_GEMI_AOD.py --proj CRC-DX --plr 1e-3 --elr 1e-3 --num_epochs 15 --smoothing 0.05 
python run_GEMI_AOD.py --proj STAD-DX --plr 1e-3 --elr 1e-3 --num_epochs 15 --smoothing 0.1
python run_GEMI_AOD.py --proj GBM-DX --plr 1e-3 --elr 1e-3 --num_epochs 50 --smoothing 0.01 
```

### Postprocessed whole-slide-image datasets

Please download from