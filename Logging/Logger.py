import logging as lg
import os


class logger:
    def __init__(self):
        path = os.getcwd() + '/' + 'Prediction_Logs/'
        # path = "..//Prediction_Logs/"
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.INFO, format='%(asctime)s %(message)s')
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.WARNING, format='%(asctime)s %(message)s')
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.ERROR, format='%(asctime)s %(message)s')
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.DEBUG, format='%(asctime)s %(message)s')
        # lg.basicConfig(filename="test.log" , format="%(asctime)s %(message)s")

    def info(self, Message):
        lg.info(Message)

    def warning(self, Message):
        lg.warning(Message)

    def error(self, Message):
        lg.error(Message)

    def sd(self):
        lg.shutdown()


def t(a,b):
    LG = logger()
    try:
        res = a/b
        return res
    except Exception as e:
        print('error is:\n' ,e)
        LG.info(str(e))
        
    
t(5,0)

# lg.shutdown()

# print(os.getcwd())
