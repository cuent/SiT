
((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py   ref_batches/VIRTUAL_imagenet256_labeled.npz   /workspace/SiT/samples/SiT-XL-2-pretrained-cfg-1.5-4-ODE-250-dopri5.npz
Computing evaluations...
Inception Score: 256.44427490234375
FID: 2.122269620697409
sFID: 4.596006894232687
Precision: 0.80796
Recall: 0.6051




((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py /workspace/SiT/references/blurred_imagenet_128_train_50k_seed0.npz /workspace/SiT/samples/fid-012-150k-interflow-dopri5/SiT-XL-2-0150000-cfg-1.0-1000-ODE-2-dopri5.npz
Computing evaluations...
Inception Score: 15.253585815429688
FID: 35.213895103772415
sFID: 8.857193086947177
Precision: 0.4231
Recall: 0.58874




((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py /workspace/SiT/references/blurred_imagenet_128_train_50k_seed0.npz /workspace/SiT/samples/fid-012-150k-interflow-dopri5/SiT-XL-2-0450000-cfg-1.0-1000-ODE-2-dopri5.npz
Computing evaluations...
Inception Score: 18.66999053955078
FID: 27.090916485883383
sFID: 8.276302559918236
Precision: 0.47596
Recall: 0.61214



((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py /workspace/SiT/references/blurred_imagenet_128_train_50k_seed0.npz /workspace/SiT/samples/fid-012-150k-interflow-dopri5/SiT-XL-2-0500000-cfg-1.0-1000-ODE-2-dopri5.npz
Computing evaluations...
Inception Score: 18.917055130004883
FID: 26.34925374302742
sFID: 8.235173973063183
Precision: 0.48204
Recall: 0.62226


((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py /workspace/SiT/references/blurred_imagenet_128_train_50k_seed0.npz /workspace/SiT/samples/fid-012-150k-interflow-dopri5/SiT-XL-2-0600000-cfg-1.0-1000-ODE-2-dopri5.npz
Computing evaluations...
Inception Score: 19.616756439208984
FID: 25.401352848102306
sFID: 8.206956159432593
Precision: 0.4871
Recall: 0.62498




((venv_guided_diff) ) root@C.35481954:/workspace/guided-diffusion/evaluations$ python evaluator.py /workspace/SiT/references/blurred_imagenet_128_train_50k_seed0.npz /workspace/SiT/samples/fid-012-150k-interflow-dopri5/SiT-XL-2-0700000-cfg-1.0-1000-ODE-2-dopri5.npz
Computing evaluations...
Inception Score: 20.121623992919922
FID: 24.745151809249933
sFID: 8.154445672028714
Precision: 0.49