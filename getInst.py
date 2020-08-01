from instalooter.looters import ProfileLooter

# Выгрузка изображений со страницы пользователя domain в папку foldername
def checkInsta(domain, foldername):
    lt = ProfileLooter(domain)
    lt.download(foldername)
