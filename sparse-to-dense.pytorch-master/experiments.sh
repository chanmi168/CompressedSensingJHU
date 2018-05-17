
set -e
set -u
set -x

# source activate cs_project

# # Experiment 1. RMSE should drop when depth samples increase (the best should be s=1000)
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 30
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 300
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 1000

# # Experiment 2. Compare performance using different decoder (the best should be deconv3)
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100
# python3 main.py -a resnet50 -d upproj -m rgbd -s 100
# python3 main.py -a resnet50 -d upconv -m rgbd -s 100


# # Experiment 4. Compare performance using input (the best should be rgbd)
# python3 main.py -a resnet50 -d deconv3 -m d -s 100
# python3 main.py -a resnet50 -d deconv3 -m rgb -s 100
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100

# # Experiment 3. Compare performance using different resnet encoder (the best should be resnet200)
# python3 main.py -a resnet18 -d deconv3 -m rgbd -s 100
# python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100
python3 main.py -a resnet152 -d deconv3 -m rgbd -s 100
