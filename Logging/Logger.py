import logging as lg
import os


class logger:
    '''
    class name : logger

    parameter : None

    work : To save different type of logs

    '''
    def __init__(self):
        path = os.getcwd() + '/' + 'Prediction_Logs/'
        # path = "..//Prediction_Logs/"
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.INFO, format='%(asctime)s %(message)s')
        # lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.WARNING, format='%(asctime)s %(message)s')
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.ERROR, format='%(asctime)s %(message)s')
        lg.basicConfig(filename=path + "Final_Logs.txt", level=lg.DEBUG, format='%(asctime)s %(message)s')
        # lg.basicConfig(filename="test.log" , format="%(asctime)s %(message)s")

    def info(self, Message):
        '''
        method name : info

        parameter : Message : message to save in logs as info

        return : None
        '''
        lg.info(Message)

    def warning(self, Message):
        '''
        method name : warning

        parameter : Message : message to save in logs as warning

        return : None
        '''
        lg.warning(Message)

    def error(self, Message):
        '''
        method name : error

        parameter : Message : message to save in logs as error

        return : None
        '''
        lg.error(Message)

    def sd(self):
        '''
        method name : error

        parameter : None

        work : specially created for shutdown logging object

        return : None
        '''
        lg.shutdown()




