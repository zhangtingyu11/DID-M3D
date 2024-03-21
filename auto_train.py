import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import re
import socks
import random
def replace_numbers(input_string, replacement_number):
    # 使用正则表达式匹配字符串中的数字
    result = re.sub(r'\d+', str(replacement_number), input_string)
    return result

def get_best_3dmod_acc(filename):
    mean_best = -1
    best = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '3d@0.70' in line:
                match = re.search(r'\[([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\]', line)
                if match:
                    # 获取匹配到的三个数值
                    easy, mod, hard = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                    mean = (easy+mod+hard)/3
                    if mean > mean_best:
                        mean_best = mean
                        best = [easy, mod, hard]
    return best

def change_yaml_name(yaml_file, index):
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'log_dir' in line:
                log_dir_content = lines[idx]
                log_dir_content = replace_numbers(log_dir_content, index)
                lines[idx] = log_dir_content
            if 'out_dir' in line:
                out_dir_content = lines[idx]
                out_dir_content = replace_numbers(out_dir_content, index)
                lines[idx] = out_dir_content
    with open(yaml_file, 'w') as f:
        f.writelines(lines)
        
def send_email(easy, mod, hard, map, res):
    # 发件人和收件人信息
    sender_email = "18013933973@163.com"
    receiver_email = "18013933973@163.com"
    password = "SNOYAHKUJNPWATEF"

    # 邮件内容
    subject = "使用LRRU的深度补全结果(去掉<2.0的深度, 增加关键点预测, 200轮, 以图像黑框为界_过滤帧, 设置权重变化起点"
    if res == 0:
        body = "当前各个难度的AP为({}, {}, {}), mAP为{}".format(easy, mod, hard, map)
    else:
        body = "程序出错啦, 当前各个难度的AP为({}, {}, {}), mAP为{}".format(easy, mod, hard, map)
        
    
    # 创建 MIMEText 对象
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = Header(subject, "utf-8")
    message["From"] = sender_email
    message["To"] = receiver_email
    # 连接到网易邮箱 SMTP 服务器
    # socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, '127.0.0.1', 20171)
    
    # socks.wrapmodule(smtplib)
    with smtplib.SMTP("smtp.163.com", 25) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

if __name__ == "__main__":
    for idx in range(0, 100):
        change_yaml_name('/home/zty/Project/DeepLearning/DID-M3D/config/kitti_car.yaml', idx)
        res = os.system('python tools/train_val.py --config config/kitti_car.yaml')
        log_filename = 'work_dirs/kitti_models/logs/only_car_lrru_clip_keypoint_right_epoch_two_hundred_以图像黑框为界_过滤帧_设置权重变化起点_{}/train.log'.format(idx)
        if res == 0:
            easy, mod, hard = get_best_3dmod_acc(log_filename)
            send_email(easy, mod, hard, (easy+mod+hard)/3, res)
            
        else:
            easy, mod, hard = 0, 0, 0
            send_email(easy, mod, hard, (easy+mod+hard)/3, res)
            break