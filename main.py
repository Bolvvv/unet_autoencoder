import torch
from torch import utils
from dataset import CustomDataset
from config import TrainingConfig
from torchvision import datasets, transforms, models
import torch.nn as nn
import time
from unet import UNet
import sys
from PIL import Image

#设置cuda使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, config, train_loader, criterion, optimizer, epoch):
    model.train()

    #显示参数设定
    total_step = len(train_loader)
    total = 0#数据集总数
    total_loss = 0#计算一个batch的所有loss
    n_iter_loss = 0#计算一个n_iter的所有loss

    for i ,(src_images, tgt_images) in enumerate(train_loader):
        src_images = src_images.to(device)
        tgt_images = tgt_images.to(device)

        #Forward
        output = model(src_images)
        loss = criterion(output, tgt_images)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #计算参数
        total += src_images.size(0)
        total_loss += loss.item()*src_images.size(0)
        n_iter_loss += loss.item()

        if (i+1) % config.show_n_iter == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.6f}'
                .format(epoch+1, config.epoches, i+1, total_step, n_iter_loss/config.show_n_iter))
            n_iter_loss = 0#展示完成后清零
    sum_loss = total_loss / total
    return sum_loss

def valid_step(model, config, valid_loader, criterion):
    model.eval()

    with torch.no_grad():
        total = 0
        total_loss = 0
        for src_images, tgt_images in valid_loader:
            src_images = src_images.to(device)
            tgt_images = tgt_images.to(device)

            #Forward
            output = model(src_images)
            loss = criterion(output, tgt_images)

            #计算参数
            total += src_images.size(0)
            total_loss += loss.item()*src_images.size(0)
        sum_loss = total_loss / total
        print('Valid loss of the model on the valid images: %.6f '% (sum_loss))

    return sum_loss

def train_valid(model, config, train_loader, valid_loader):
    #记录最低的Loss
    min_loss = sys.maxsize
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    result_file_path = './result/'+time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))+'_results.csv'
    #将信息写入csv文件中
    with open(result_file_path, 'w') as f:
        f.write('batch_size %d, lr %f, epoches %d, start_time %s\n' % (config.batch_size, config.lr, config.epoches, time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))))
        f.write('epoch,train_loss,valid_loss\n')
    
    for epoch in range(config.epoches):
        train_loss = train_step(model, config, train_loader, criterion, optimizer, epoch)
        valid_loss = valid_step(model, config, valid_loader, criterion)
        if valid_loss < min_loss:
            min_loss = valid_loss
            print("New min loss: %.6f" % min_loss)
            torch.save(model.state_dict(), './result/model.dat')
        with open(result_file_path, 'a') as f:
            f.write('%03d,%0.6f,%0.6f\n' % (
                (epoch + 1),
                train_loss,
                valid_loss
            ))
    with open(result_file_path, 'a') as f:
        f.write('End Time %s\n' % (time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))))

def load_data(config):
    mean = [0.5,]
    stdv = [0.2,]
    src_transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    tgt_transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    train_set = CustomDataset(filename="../data/deepbc/autoencoder_labels/train.txt", src_dir="../data/deepbc/usg_images_cutted_p1", tgt_dir="../data/deepbc/usg_images_cutted_v3", src_transform=src_transform, tgt_transform=tgt_transform)
    valid_set = CustomDataset(filename="../data/deepbc/autoencoder_labels/valid.txt", src_dir="../data/deepbc/usg_images_cutted_p1", tgt_dir="../data/deepbc/usg_images_cutted_v3", src_transform=src_transform, tgt_transform=tgt_transform)
    test_set = CustomDataset(filename="../data/deepbc/autoencoder_labels/test.txt", src_dir="../data/deepbc/usg_images_cutted_p1", tgt_dir="../data/deepbc/usg_images_cutted_v3", src_transform=src_transform, tgt_transform=tgt_transform)

    #载入训练参数
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()))

    return train_loader, valid_loader, test_loader

#生成单张图片
def generate_img(model, img_path, out_img_path):
    #模型载入
    model.load_state_dict(torch.load('./result/model.dat'))
    #tensor和PIL相互转化
    loader = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    
    img = Image.open(img_path)#PIL图片
    tensor_img = loader(img).unsqueeze(0).to(device)#转换为tensor
    out = model(tensor_img)#输出图片
    pil_img = unloader(out.squeeze(0))#转换为PIL图片
    pil_img.save(out_img_path)

if __name__ == "__main__":
    #获取训练超参
    config = TrainingConfig
    #载入数据
    train_loader, valid_loader, test_loader = load_data(config)
    #初始化模型
    model_unet = UNet(3, 3, bilinear=True)
    model_unet = model_unet.to(device)

    if config.train:
        train_valid(model_unet, config, train_loader, valid_loader)

    # generate_img(model_unet, "0913_1_1.jpg", "hhh.jpg")