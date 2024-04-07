CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 666632 main.py --train_batch_size 20 --dataset dataset/VisDA --name visda --source_list dataset/VisDA/source.txt --target_list dataset/VisDA/target.txt --test_list dataset/VisDA/target.txt --num_classes 12 --model_type ViT-B_16 --pretrained_dir checkpoint2/ViT-B_16.npz --num_steps 20000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im
CUDA_VISIBLE_DEVICES=1 python3 T-sne.py --dataset office --name aw --num_classes 31 --image_path data/office/amazon_list.txt  --image_path2 data/office/webcam_list.txt --img_size 256
#CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 66661 main.py --train_batch_size 20 --dataset dataset/VisDA --name visda --source_list dataset/VisDA/source.txt --target_list dataset/VisDA/target.txt --test_list dataset/VisDA/target.txt --num_classes 12 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 20000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im
# 3GPUS, patchmix , no mixloss:84.91
# 3GPUS, patchmix , mixloss :85.27
# 3GPUS, patchmix , mixloss, HD:85.26
# 85.4

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 66661 main.py --train_batch_size 20 --dataset DomainNet --name PS --source_list /home/cs23-tancz/TVT-mixup+H/data/DomainNet/Painting/painting_train1.txt --target_list /home/cs23-tancz/TVT-mixup+H/data/DomainNet/Sketch/sketch_train1.txt --test_list /home/cs23-tancz/TVT-mixup+H/data/DomainNet/Sketch/sketch_test1.txt --num_classes 345 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 200000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im


python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 66661 main.py --train_batch_size 20 --dataset DomainNet --name PS --source_list data/DomainNet/Painting/painting_train2.txt --target_list data/DomainNet/Sketch/sketch_train2.txt --test_list data/DomainNet/Sketch/sketch_test2.txt --num_classes 345 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 200000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 66661 main.py --train_batch_size 20 --dataset DomainNet --name SP --source_list data/DomainNet/Sketch/sketch_train2.txt --target_list data/DomainNet/Painting/painting_train2.txt --test_list data/DomainNet/Painting/painting_test2.txt --num_classes 345 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 200000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im

CUDA_VISIBLE_DEVICES=2 python3 main.py --train_batch_size 32 --dataset RI --name RI --source_list /home/cs23-tancz/TVT/data/DomainNet/Quickdraw/quickdraw_train1.txt --target_list /home/cs23-tancz/TVT/data/DomainNet/Infograph/infograph_train1.txt --test_list /home/cs23-tancz/TVT/data/DomainNet/Infograph/infograph_test1.txt --num_classes 345 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 200000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im --lamda 0.1

CUDA_VISIBLE_DEVICES=2 python3 main.py --train_batch_size 64 --dataset RI --name RI --source_list /home/cs23-tancz/TVT/data/DomainNet/Quickdraw/quickdraw_train1.txt --target_list /home/cs23-tancz/TVT/data/DomainNet/Infograph/infograph_train1.txt --test_list /home/cs23-tancz/TVT/data/DomainNet/Infograph/infograph_test1.txt --num_classes 345 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 200000 --img_size 256 --beta 1.0 --gamma 0.01 --use_im --lamda 0.1

