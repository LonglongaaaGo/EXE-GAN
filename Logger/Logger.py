




class Logger:
    def __init__(self,path,continue_=True):
        self.path = path
        self.continue_ = continue_
        if continue_ == True:
            self.loger = open(path,mode="a+")
        else:
            self.loger = open(path,mode="w+")
        self.loger.close()

    def update(self,iter,**kwargs):
        if self.continue_ == True:
            self.loger = open(self.path, mode="a+")

        out_line = f"[{str(iter).zfill(7)}]\t"
        for key in kwargs:
            out_line+=f"[{key}]:{kwargs[key]}\t"
        out_line+="\n"
        print(out_line)
        self.loger.write(out_line)

        self.loger.close()




