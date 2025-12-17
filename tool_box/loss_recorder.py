import pandas as pd
import matplotlib.pyplot as plt

class LossRecorder():
    def __init__(self,col=['tag','step','loss','pde','bc','anc']) -> None:
        # self.path = storePath
        self.loss_data = pd.DataFrame(columns=col)
        self.loss_datas=[]

    def append(self,datas):
        self.loss_data.loc[len(self.loss_data)] = datas

    def print(self):
        self.loss_data.describe()

    def save_as_csv(self,sp="loss.csv"):
        self.loss_data.to_csv(sp)

    def load_csv(self,path):
        self.loss_data = pd.read_csv(path,index_col=0)

    def load_multiple_csv(self,paths):
        for p in paths:
            self.loss_datas.append(pd.read_csv(p,index_col=0))

    def plot_multiple(self):
        if self.loss_datas.__len__() == 0: return
        plt.figure(figsize=(8, 6))
        index = 1
        for ls in self.loss_datas:
            x = ls.index  # The DataFrame's index (default is 0, 1, 2, 3, ...)
            y = ls['L2']  # The values from the 'values' column
            # plt.plot(x, y, label=f'L2 at {index}', marker='o', color='b')
            plt.plot(x, y, label=f'L2 at {index}')
            index = index + 1
        plt.xlabel('epoch')
        plt.ylabel('l2')
        plt.grid()
        plt.legend()
        plt.show()

    def plot(self,path='loss.png'):
        x = self.loss_data.index  # The DataFrame's index (default is 0, 1, 2, 3, ...)
        y = self.loss_data['L2']  # The values from the 'values' column
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label='L2', marker='o', color='b')
        plt.title('L2 loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        # plt.show()
        plt.savefig(path)

if __name__ == '__main__':
    lr = LossRecorder()
    csvs = []
    for i in range(1,8):
        csvs.append(f'../loss_record/loss-slant-re600-dtau{i}.00e+00.csv')
        # csvs.append(f'../loss_record/loss-slant-re600-dtau{i}.00e-01.csv')
    lr.load_multiple_csv(csvs)
    lr.plot_multiple()
    # lr.load_csv('../loss_record/loss-slant-re600-dtau1.00e-01.csv')
    # lr.plot()
