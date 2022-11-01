#DINO
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill dino --beta 1.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill dino --beta 2.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill dino --beta 5.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill dino --beta 10.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill dino --beta 100.0
#CRD
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --beta 1.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --beta 2.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --beta 5.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --beta 10.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --beta 100.0
