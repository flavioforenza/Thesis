import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import MobileNetV1_Teach, MobileNetV1_Stud

CHECKPOINTS_DIR = './checkpoints'
MODEL_DIR = './model'

resolution=224
alpha=0.25

def evaluate(model_type='student', prefix='student'):
    if model_type == 'teacher':
        model = MobileNetV1_Teach(8)
    elif model_type == 'student':
        model = MobileNetV1_Stud(8, alpha)
       
    model_paths = []
    for epoch_count in range(0, 1000, 1):
        model_paths.append(os.path.join(CHECKPOINTS_DIR, '{}-{}.pth'.format(prefix, epoch_count+1)))
    #model_paths.append(os.path.join(MODEL_DIR, '{}.pth'.format(prefix)))
    
    valdir = os.path.join('./data/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    batch_size = 8

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    error_counts = np.zeros(shape=len(model_paths), dtype=np.int32)
    for model_count, model_path in enumerate(model_paths):
        md = torch.load(model_path)
        model.load_state_dict(md.state_dict())
        model.eval()
        model = model.cuda()

        error_count = 0
        gpu=0
        for _, (images, target) in enumerate(data_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)
            # compute output
            output = model(images)
            #loss = criterion(output, target)
            with torch.no_grad():
                output = torch.argmax(output, dim=1)
                error_count += batch_size - torch.eq(output, target).sum()
        
        error_counts[model_count] = error_count
        print('{}-{}: {}'.format(prefix, model_count, error_count))

    return error_counts

# Evaluate accuracy
#error_counts_teacher = evaluate('teacher', 'teacher')
#error_counts_student = evaluate('student', 'student_distill')

# Store as file
#np.save('./results/error_counts_teacher_normal2.npy', error_counts_teacher)
#np.save('./results/error_counts_student_distill__0.25_224_t=4.npy', error_counts_student)

# # Load from file
error_counts_teacher = np.load('./results/final_error_counts_teacher_normal.npy')
#error_counts_student = np.load('./results/error_counts_student_conv2D_half.npy')
#error_counts_student_2 = np.load('./results/error_counts_student_conv2D_half_NoBatch.npy')
error_counts_student_3 = np.load('./results/error_counts_student_0.25_224.npy')
#error_counts_student_4 = np.load('./results/error_counts_student_0.25_128.npy')
error_counts_student_distill_0_24_224_t1 = np.load('./results/error_counts_student_distill__0.25_224_t=1.npy')
error_counts_student_distill_0_24_224_t2 = np.load('./results/error_counts_student_distill__0.25_224_t=2.npy')
error_counts_student_distill_0_24_224_t3 = np.load('./results/error_counts_student_distill__0.25_224_t=3.npy')
error_counts_student_distill_0_24_224_t4 = np.load('./results/error_counts_student_distill__0.25_224_t=4.npy')
error_counts_student_distill_0_24_224_t5 = np.load('./results/error_counts_student_distill__0.25_224_t=5.npy')
error_counts_student_distill_0_24_224_t10 = np.load('./results/error_counts_student_distill__0.25_224_t=10.npy')
error_counts_student_distill_0_24_224_t15 = np.load('./results/error_counts_student_distill__0.25_224_t=15.npy')

print("Minimo errore: ", min(error_counts_student_distill_0_24_224_t4))
print("Massimo errore: ", max(error_counts_student_distill_0_24_224_t4))

# Prepare to plot
fig, ax = plt.subplots(1,1,figsize=(16,7))

step = 250
lst_x=[1]
lst_idx = [0]
for x in range(0, 1001, step):
    if x!=0:
        lst_idx.append(x-1)
        lst_x.append(x)

lst_y_teach = [error_counts_teacher[x] for x in lst_idx]
print(len(lst_y_teach))

#lst_y_stud = [error_counts_student[x-1] for x in range(1,1001, 199)]
#lst_y_stud_2 = [error_counts_student_2[x-1] for x in range(1,1001, 199)]

lst_y_stud_3 = [error_counts_student_3[x] for x in lst_idx]
#lst_y_stud_4 = [error_counts_student_4[x-1] for x in range(1,1001, 199)]
lst_y_stud_5 = [error_counts_student_distill_0_24_224_t1[x] for x in lst_idx]
lst_y_stud_6 = [error_counts_student_distill_0_24_224_t5[x] for x in lst_idx]
lst_y_stud_7 = [error_counts_student_distill_0_24_224_t10[x] for x in lst_idx]
lst_y_stud_8 = [error_counts_student_distill_0_24_224_t15[x] for x in lst_idx]
lst_y_stud_9 = [error_counts_student_distill_0_24_224_t2[x] for x in lst_idx]
lst_y_stud_10 = [error_counts_student_distill_0_24_224_t3[x] for x in lst_idx]
lst_y_stud_11 = [error_counts_student_distill_0_24_224_t4[x] for x in lst_idx]

print(len(lst_y_stud_10))

ax.plot(lst_x, lst_y_teach, label='Teacher', linewidth=5)
#ax.plot(lst_x, lst_y_stud, label='std_C2DH_loss')
#ax.plot(lst_x, lst_y_stud_2, label='std_C2DHNB_loss')
ax.plot(lst_x, lst_y_stud_3, label='Student_0.25_224', linewidth=5)
#ax.plot(lst_x, lst_y_stud_4, label='std_0.25_128loss')
ax.plot(lst_x, lst_y_stud_5, label='Student_0.25_224_T=1', alpha=0.5)
ax.plot(lst_x, lst_y_stud_9, label='Student_0.25_224_T=2',alpha=0.5)
ax.plot(lst_x, lst_y_stud_10, label='Student_0.25_224_T=3', linewidth=5)
ax.plot(lst_x, lst_y_stud_11, label='Student_0.25_224_T=4', alpha=0.5)
ax.plot(lst_x, lst_y_stud_6, label='Student_0.25_224_T=5', alpha=0.5)
ax.plot(lst_x, lst_y_stud_7, label='Student_0.25_224_T=10', alpha=0.5)
ax.plot(lst_x, lst_y_stud_8, label='Student_0.25_224_T=15',alpha=0.5)

#ax.plot(range(0, 3001, 100), error_counts_teacher_distill, label='teacher with distillation')

ax.set_xlabel('number of epochs')
ax.set_ylabel('number of errors')
ax.set_title('Learning curve')


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=r'$\bf{Models}$')
line = leg.get_lines()
line[4].set_linewidth(4.0)
plt.show()

name_img = 'student_distill_0.25_224_T4'
plt.savefig('./images/'+name_img+'.png')