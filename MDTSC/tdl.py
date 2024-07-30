# tensor base learning
# 基于张量处理优化频域内数据
import scipy.optimize as sco
import numpy as np
from parm import Para as P
# FFT:快速傅里叶变换 IFFT:反快速傅里叶变换

def tendl(Xhat,S,P): # 输入数据的FFT变换结果，另一组数据，参数实例
    r = P.r
    Shat = np.fft.fft(S,axis = -1)
    dual_lambda = 10*abs(np.random.randn(r)) # 优化过程中的对偶变量
    m,_,k = Xhat.shape
    SSt = np.zeros((r,r,k))
    SSt = np.fft.fft(SSt,axis = -1)
    XSt = np.zeros((m,r,k))
    XSt = np.fft.fft(XSt,axis = -1)
    for kk in range(k): # 遍历每个频率
        xhatk = Xhat[:,:,kk]
        shatk = Shat[:,:,kk]

        SSt[:,:,kk] = np.dot(shatk,np.conj(shatk).T)
        XSt[:,:,kk] = np.dot(xhatk,np.conj(shatk).T)

    # optimise return x
    bnds = tuple((0,np.infty) for i in range(len(dual_lambda))) # 边界，保证对偶变量非负
    fun = lambda x :fobj(x,XSt,SSt,k)
    res = sco.minimize(fun,dual_lambda,method = 'L-BFGS-B',bounds = bnds) # 求解最优化问题

    Lambda = np.diag(res.x)
    Bhat = np.zeros((m,r,k))
    Bhat = np.fft.fft(Bhat,axis = -1)
    for kk in range(k):
        SStk = SSt[:,:,kk]
        XStk = XSt[:,:,kk]
        Bhatkt =np.dot(np.linalg.pinv(SStk+Lambda),np.conj(XStk).T)
        Bhat[:,:,kk] = np.conj(Bhatkt).T

    B = np.fft.ifft(Bhat,axis = -1) # 用IFFT得到结果B，清理NANs并将数据转换为实数
    B[np.where(np.isnan(B) == True)] = 0
    B = np.real(B)

    return B

def fobj(lam,XSt,SSt,k):
    m = XSt.shape[0]
    r = np.size(lam)
    Lam = np.diag(lam)
    f = 0
    for kk in range(k): # 计算每个频段的负对数似然
        XStk = XSt[:,:,kk]
        SStk = SSt[:,:,kk]
        SSt_inv = np.linalg.pinv(SStk+Lam)
        if m>r:
            f = f+np.trace(np.dot(SSt_inv,np.dot(np.conj(XStk).T,XStk)))
        else :
            f = f+np.trace(np.dot(np.dot(XStk,SSt_inv),np.conj(XStk.T)))
    f = np.real(f+k*sum(lam))

    return f

if __name__ == '__main__':
    ss = np.random.rand(30,33614,25)
    xx = np.random.rand(25,33614,25)
    xx_h = np.fft.fft(xx,axis=-1)
    print(tendl(xx_h,ss,P).shape)

