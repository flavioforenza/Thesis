
from mobilenet_ssd import create_mobilenetv1_ssd
from train_sd import train_ssd

ssd_mobilenet_dist = create_mobilenetv1_ssd(8, './checkpoints/student_distill-1000.pth', 0.25)
#ssd_mobilenet_dist.save('./model/ssd_distilled.pth')



print("End")