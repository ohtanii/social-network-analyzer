import os.path 
import getVK as vk
import getInst as inst
import imgClass as ic
import sentimentAnalysis as sa

vk_domain = input ('Введите id или domain пользователя Вконтакте (для пропуска введите 0): \n')
inst_domain = input ('Введите имя пользователя Instagram (для пропуска введите 0): \n')

if ((vk_domain=='0') & (inst_domain=='0')):
    print ("\nДанные для поиска не были введены")
    exit()

foldername = input('Введите название папки для данного поиска: \n')
if not os.path.exists(foldername):
	os.mkdir(foldername)

if not(vk_domain=='0'):
    vk.scanVK(vk_domain, foldername)

if not(inst_domain=='0'):
    inst.checkInsta(inst_domain,foldername)

ic.imgModule(foldername)
sa.sentimentAn(foldername)