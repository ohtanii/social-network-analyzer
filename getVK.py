#import vk_api
import requests
import csv
import random 
import os.path 

domain = ''
token = '90f52caf90f52caf90f52caf7390840e04990f590f52cafce6dad2f54b03da5786280f5'
version = 5.103

name = ''
bdate = ''
sex = 0
city = ''
isPrivate = False

# Выгрузка всех записей со стены
def getAllPosts(domain):
	# Максимальное количество записей для одной выгрузки - 100; начальный сдвиг - 0
	count = 100
	offset = 0
	allPosts = []

	# Получение записей по 100 за запрос, пока значение сдвига не станет больше или равным общему числу записей на стене
	while True:
		response = requests.get('https://api.vk.com/method/wall.get', params={'access_token': token, 'v': version, 'domain': domain, 'count': count, 'offset': offset}) 
		try:
			data = response.json()['response']['items']
		except:
			print ('\n ERROR occurred.', response.json()['error']['error_msg'])
			break
		allPosts.extend(data)
		if offset>= response.json()['response']['count']:
			break
		offset+=100
	return allPosts

# Выгрузка записей на стене и количества лайков в файл формата VK_*domain*/posts.csv
def postsToFile(data, foldername):
	with open(foldername+'/posts.csv','w') as file:
		a_pen = csv.writer(file)
		a_pen.writerow(('description', 'likes'))
		for post in data:
			a_pen.writerow((post['text'], post['likes']['count']))

# Получение основной информации о странице
def getInfo(domain, foldername):
	response = requests.get('https://api.vk.com/method/users.get', params={'access_token': token, 'v': version, 'user_ids': domain, 'fields': 'photo_max, city, sex, bdate, has_photo, is_closed'})
	# Проверка на получение сообщения об ошибке
	try:
		response.json()['response']
	except Exception:
		print('\nПроизошла ошибка. Причина: ', response.json()['error']['error_msg'])
		return False

	# Проверка на наличие фото профиля и его загрузкапри наличии
	if response.json()['response'][0]['has_photo']:
		filename=foldername+'/pic_VK.jpg'
		profilePic = response.json()['response'][0]['photo_max']
		with open(filename, 'wb+') as handle:
			resp = requests.get(profilePic, stream=True)
			handle.write(resp.content)
	global name
	# Выгрузка имени
	name = str(response.json()['response'][0]['first_name'])+ ' ' +str(response.json()['response'][0]['last_name'])
	# Проверка, закрытый ли профиль
	if response.json()['response'][0]['is_closed']:
		if not response.json()['response'][0]['can_access_closed']:
			global isPrivate
			isPrivate = True
	# Выгрузка указанного пола, даты рождения и города
	global sex
	try:
		sex = response.json()['response'][0]['sex']
	except:
		pass
	global bdate
	try:
		bdate = response.json()['response'][0]['bdate']
	except:
		pass
	global city
	try:
		city = response.json()['response'][0]['city']['title']
	except:
		pass
	return True

def scanVK(domain, foldername):
	if getInfo(domain, foldername):
		print('\n       Информация о пользователе:','\n\nИмя: ', name)
		if bdate:
			print('Дата рождения: ', bdate)
		if sex==1:
			print('Пол: женский')
		elif sex==2:
			print('Пол: мужской')
		else:
			print('Пол: не указан')
		if city:
			print('Город: ', city)
		if isPrivate:
			print('Профиль является закрытым. Невозможно получить больше информации.')
		
	if not isPrivate:
		allPosts = getAllPosts(domain)
		if allPosts:
			postsToFile(allPosts, foldername)
		else:
			print('Нет доступных записей на стене.')