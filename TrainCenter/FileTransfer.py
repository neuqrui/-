import os
import xmlrpc.client
from Storage import Log


class FileTransfer:
    @staticmethod
    def download_data(folder, data, file_name, part, format_name):
        """
        Multi-threaded Download files from the client
        :param folder:(str)
        :param data:(file)
        :param file_name:(str)
        :param part:int
        :param format_name:(str)
        :return:None
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        handle = open(folder + file_name + "_{}".format(part) + format_name, 'wb')
        handle.write(data.data)
        handle.close()
        print('part {} created......'.format(part))

    @staticmethod
    def merge_files(folder, filename, part_list, format_name):
        """
        Multi-threaded Combine file fragments
        :param folder:(str)
        :param filename:(str)
        :param part_list:(list)
        :param format_name:(str)
        :return:None
        """
        outfile = open(folder + filename + format_name, 'wb')
        for part in part_list:
            file = folder + filename + '_' + str(part) + format_name
            infile = open(file, 'rb')
            data = infile.read()
            outfile.write(data)
            infile.close()
        outfile.close()
        Log.print_string("merge " + filename + " finished!")
        
    @staticmethod
    def upload_model(model_path):
        handle = open(model_path, 'rb')
        return xmlrpc.client.Binary(handle.read())
    
    @staticmethod
    def download_iter_train_data(folder, data, file_name, format_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        handle = open(folder + file_name + format_name, 'wb')
        handle.write(data.data)
        handle.close()
        Log.print_string("iter train data received")
