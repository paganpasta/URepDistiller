pip install -r requirements.txt

python -m torch.distributed.launch --nproc_per_node=1  main.py --local_rank=0 -a resnet18 -k resnet50 --teacher_ssl moco --method coss --t-temp 0.02 --s-temp 0.7 --beta 50.0 \
--output ../outputs/students/imagenet/resnet50/resnet18/coss --distill ../outputs/teachers/imagenet/seed/moco_v2.pth --data ../data/imagenet/Data/CLS-LOC/

