# the para which will be used
# 噪声：图像中不需要或额外的信息，使视觉上模糊or粗糙

class Para():
    patsize = 3
    step = 2 # 步长
    r = 30 # 维度
    eta = 1.01 # 调整因子
    maxiter = 10 # 最大迭代次数
    maxiterB = 50 
    denoiseiter = 10 # 去噪迭代次数
    beta = 0.8 # 衰减因子？
