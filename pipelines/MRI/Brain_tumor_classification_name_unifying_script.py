import glob
import os
if __name__ == '__main__':
    glioma = []
    meningioma = []
    good = []
    pituitary = []
    #put the directory here
    dir_to_new_db = r''
    #searching for pngs in our database
    for item in glob.glob("db\\glioma_tumor\\*.png"):
        glioma.append(item)
    for item in glob.glob("db\\meningioma_tumor\\*.png"):
        meningioma.append(item)
    for item in glob.glob("db\\no_tumor\\*.png"):
        good.append(item)
    for item in glob.glob("db\\pituitary_tumor\\*.png"):
        pituitary.append(item)

    #here we iterate over files to unify the names
    #and to put them in new directory
    for i,item in enumerate(glioma):
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Glioma.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(meningioma):
        i = i+len(glioma)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Meningioma.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(good):
        i = i + len(glioma) +len(meningioma)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Good.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(pituitary):
        i = i + len(glioma) + len(meningioma) + len(pituitary)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Pituitary.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
